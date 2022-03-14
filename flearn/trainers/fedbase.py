import numpy as np
import random
import os
import json
import math
from utils.poisson_test import draw_bar_poisson
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
from scipy.stats import beta
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn import cluster

INITIAL_EPOCH = 0.1

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);                          #将params中的参数都设给self,则可以通过self.seed获得参数值

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)   #learner是mclr这种，*params['model_params']是mclr,inner_opt是优化器，和seed都是子类参数
        self.clients = self.setup_clients(dataset, self.client_model)                     #clients代表了dataset里所有的Client类，类中包括id,group,train_data,test_data
        self.initial_actual_epoch(self.clients, params['num_epochs'])                     #使用传入的命令行参数作为初始分配epoch
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()                                #返回的是模型的参数

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)                                      #创建一个Metrics实例用来评估模型

        # create Pk to represents selected probabilites
        self.Pk = self.get_probability()                                                  #创建Pk来表示客户端被选中的概率
        # create avg_loss to represents the averagy loss of selected clients
        self.avg_loss = 0
        # initial number of clients to be selected for training each round
        self.clients_per_round = params['clients_per_round']

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):                    #实例化数据集dataset中的所有Client
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset               #dataset为model_utils中read_data的返回值,clients[],groups[],train_data{},test_data{}
        if len(groups) == 0:
            groups = [None for _ in users]                           #for u, g in zip(users,grops)是把users和groups对应项取出来，这种方式比遍历users[0],group[0]方便
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)] #Client()接收的第一个参数第整型，但是user中存放的是字符串？？？
        return all_clients

    def train_error_and_loss(self):                                  #获得所有客户端训练正确样本数目，损失等
        num_samples = []                                             #存放每个客户端的样本总数
        tot_correct = []                                             #存放每个客户端分类正确的样本数
        losses = []                                                  #存放每个客户端的损失
        client_accs = []                                             #添：存放每个客户端的训练精度acc

        for c in self.clients:                                       #调用Client类的train_error_and_loss函数
            #调用Client类的train_error_and_loss()，该函数又调用mclr等模型中的test()真正开始训练
            ct, cl, ns = c.train_error_and_loss()                    #获得客户端c分类正确的样本数ct，损失cl，样本总数ns
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
            client_accs.append(ct*1.0 / ns)
        
        ids = [c.id for c in self.clients]                           #获得每个客户端的id
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses, client_accs
    '''
    #函数重载：没什么用
    def train_error_and_loss(self, indices, round=0):                                  #获得所有客户端训练正确样本数目，损失等
        num_samples = []                                             #存放每个客户端的样本总数
        tot_correct = []                                             #存放每个客户端分类正确的样本数
        losses = []                                                  #存放每个客户端的损失

        #如果round=0则说明是最开始的一轮，此时初始化所有客户端的loss
        if(round == 0):
            #调用Client类的train_error_and_loss函数
            for c in self.clients:                                       
                #调用Client类的train_error_and_loss()，该函数又调用mclr等模型中的test()真正开始训练
                ct, cl, ns = c.train_error_and_loss()                    #获得客户端c分类正确的样本数ct，损失cl，样本总数ns
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            ids = [c.id for c in self.clients]                           #获得每个客户端的id
            groups = [c.group for c in self.clients]
        #如果round是正式开始训练的轮，那么只有被选中的客户端才会训练
        else:
            for id in indices:
                ct, cl, ns = self.clients[id].train_error_and_loss()
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            ids = indices                           #获得每个客户端的id
            groups = [self.clients[id].group for id in indices]

        return ids, groups, num_samples, tot_correct, losses
    '''

    def show_grads(self):                                           #获得所有客户端和全局梯度
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []                                      #前client_num个数存放的是客户端的梯度，最后一个存放的是全局平均梯度
        samples = []

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)           
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)      #np.add()数组元素对应相加，这里更新全局梯度和
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))          #计算平均梯度，简单平均非加权平均
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):  
        # fedavg调用父类test(),此时Server子类在初始化时已经传参给fedbase初始化了父类，传的dataset使父类实例化了其中所有的clients
        # clients在实例化的时候传的参数model就是父类中的client_model
        # 然后父类test()先设置client_model是latest_model然后调用Client的test(),该函数使用model.test()带了模型参数的mclr等模型训练数据
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        client_accs = []              #记录每个客户端的测试精度acc，方便计算variance
        self.client_model.set_params(self.latest_model)                          #将client_model设为latest_model因为此处调用Client的test(),而该test是调用的mclr这种Model的test，即model.test(),此处的model必须参数指定
        for c in self.clients:
            ct, ns = c.test()                                                    #ct测试集分类正确样本数，ns测试集样本总数
            tot_correct.append(ct*1.0)                                           #client_model改变了以后self.client的model会跟着一起变化
            num_samples.append(ns)
            c.acc = ct*1.0 / ns                                                  #更新client的精度
            client_accs.append(ct*1.0 / ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        print("in fedbase num_samples: ", sum(num_samples), " tot_correct: ", sum(tot_correct))
        return ids, groups, num_samples, tot_correct, client_accs                             #返回客户端id，group，样本数，分类正确样本数，要对应使用的时候加zip()即可获得对应的参数

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):                             #挑选客户端
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        
        # print("P: ",self.Pk)
        # print("sum(Pk): ",sum(self.Pk))
        client_range = len(self.clients) #只选择前200个客户端参与训练，可选值还有len(self.clents)
        num_clients = min(num_clients, client_range)                                  #客户端数目在传进来的数目和仅有的客户端数目中取最小值，防止想取的超过已有的
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round，使用预先设定的随机数种子那么再调用将会生成相同的随机数
        #谨慎使用random.choice()
        # indices = np.random.choice(range(client_range), num_clients, p=np.array(self.Pk).ravel(), replace=False)
        # print("sum Pk", sum(self.Pk))
        # print("self.Pk: ", self.Pk)


        # print(_ids)
        # indices = np.random.choice(range(client_range), num_clients, replace=False)   #从clients下标数组中随机取出num_clients个数字组成数组并返回，不能去相同的
        # 0.2的概率随机选，0.8的概率基于loss选
        # if(np.random.random()<0.5):
        #     indices = np.random.choice(range(client_range), num_clients, replace=False)   #从clients下标数组中随机取出num_clients个数字组成数组并返回，不能去相同的
        # else:
        #     indices = np.random.choice(range(client_range), num_clients, p=np.array(self.Pk).ravel(), replace=False)

        #前50轮基于loss选择客户端，后150随机选择客户端
        #50轮基于loss，50轮随机交替,修改为每20轮交替,每10轮 math.ceil(round/100)%2 == 1
        
        if(round <= -1):
            indices = np.random.choice(range(client_range), num_clients, p=np.array(self.Pk).ravel(), replace=False)
            print("sum Pk", sum(self.Pk))
            print("self.Pk: ", self.Pk)
        else:
            indices = np.random.choice(range(client_range), num_clients, replace=False)
        
        #给选中的客户端设置能完成的epoch数
        cepoch = self.client_epoch(indices, round)
        print("epochs: ", cepoch)
        _clients = np.array(self.clients)[indices]
        for (_c,_e) in zip(_clients,cepoch):                                               #调用client类中的set_epoch()函数设置该客户端能完成的epoch数
            _c.set_ideal_epoch(_e)

        return indices, _clients
    
    # 给选中的客户端设置能完成的epoch数
    def set_epoch_for_selected_clients(self, indices, round):
        cepoch = self.client_epoch(indices, round)
        print("epochs: ", cepoch)
        _clients = np.array(self.clients)[indices]
        for (_c,_e) in zip(_clients,cepoch):                                               #调用client类中的set_epoch()函数设置该客户端能完成的epoch数
            _c.set_ideal_epoch(_e)
        
        return indices, _clients

    # 随机选一部分->根据上一轮完成的epoch聚类->使用loss选
    def hierarchical_select_clients(self, round, num_clients=20):
        
        print("hierarchical select clients\n")
        # 固定随机数种子
        np.random.seed(round)

        """随机选择一部分客户端"""
        # 随机选择一部分客户端，具体选多少呢
        client_range = len(self.clients)
        """随机选择num_clients的多少倍"""
        #尝试num_clients的三倍
        num_class = 6
        ran_num_clients = min(num_clients*num_class, client_range)
        
        #按照随机在线客户端为总数的多少
        # active_rate = 0.5
        # ran_num_clients = int(client_range * active_rate)

        ran_indices = np.random.choice(range(client_range), ran_num_clients, replace=False)

        """根据随机选中的这部分客户端的epoch聚类"""
        # 传入随机选择的客户端，last=True代表使用上一轮实际运行的epoch数量聚类，false表示使用预测的easy的epoch聚类
        # 返回值是随机一个聚类簇的clients
        hierarchical_clients_indices = self.hierarchical_clustering(ran_indices, last=False)

        """对返回的聚类结果再进行一次根据loss挑选客户端"""
        num_clients = min(num_clients, client_range)                       #防止想随机取的客户端超过客户端总数
        # 获得聚类结果下标对应的概率下标,这部分的probability列表加起来和必须是1,所以还要进行标准化
        hierarchical_probability = np.array(self.Pk)[hierarchical_clients_indices]
        initial_hierarchical_probability = np.array(hierarchical_probability) / np.float(sum(hierarchical_probability))
        bloss_indices = np.random.choice(hierarchical_clients_indices, num_clients, p=initial_hierarchical_probability.ravel(), replace=False)
        print("hierarchical_probability: ", hierarchical_probability, "initial_hierarchical_probability: ", initial_hierarchical_probability)
        print("\n")

        # 给选中的客户端设置能完成的epoch数目，接收参数为这部分clients的下标
        indices, _clients = self.set_epoch_for_selected_clients(bloss_indices, round)

        return indices, _clients

    #code from q-ffl
    def select_clients_by_samples(self, round, pk, held_out=None, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            indices: an array of indices
            self.clients[]
        '''
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round+4)
        sampling = 2   #控制分布(1:均匀采样，加权平均 2:权重采样，样本平均)

        
        if held_out: # meta-learning
            can_be_chosen = len(self.clients) - held_out
        else:
            can_be_chosen = len(self.clients)


        if sampling == 1:  # uniform sampling + weighted average
            indices = np.random.choice(range(can_be_chosen), num_clients, replace=False)
            return indices, np.asarray(self.clients)[indices]
                   
        elif sampling == 2:  # weighted sampling + simple average
            num_samples = []
            for client in self.clients[:can_be_chosen]:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(can_be_chosen), num_clients, replace=False, p=pk)
            
            #给选中的客户端设置能完成的epoch数
            cepoch = self.client_epoch(indices, round)
            #后面这里看着像写重了
            _clients = np.array(self.clients)[indices]
            for (_c,_e) in zip(_clients,cepoch):                                               #调用client类中的set_epoch()函数设置该客户端能完成的epoch数
                _c.set_ideal_epoch(_e)
            return indices, _clients
        
    def aggregate(self, wsolns):                            #wsolns存放每个客户端训练的权重
        total_weight = 0.0                                  #如果w是本地样本数的话那total_weight应该是所有客户端的样本数
        # print("len(wsoln[0][1]: ", len(wsolns[0][1]))       #wsolns[0][1]=2,看了几个都是2
        # print("wsolns: ", wsolns)
        # print("wsolns[0][1]: ", wsolns[0][1])               #wsolns[0][1]是第0个客户端的[array(),array()]
        base = [0]*len(wsolns[0][1])                        # wsolns中存放的是soln，形状是[(45,[array(),array()]),(24,[array(),array()])],里面有两个array，都是float32类型的
        # print("base: ",base)
        #print("wsolns: ", wsolns)
        for (w, soln) in wsolns:                            # w is the number of local samples，应该是把wsolns的每个元素拆分成了(w, soln),和 for ws in wsolns:类似
            print("w:", w)
            total_weight += w                               #total_weight计算总的样本数
            for i, v in enumerate(soln):                    #enumerate()用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和下标，i是参数下标，v是soln的元素，即可训练变量的值                                  
                base[i] += w*v.astype(np.float64)           #base[i]存放第i个可训练变量的值，即把所有客户端的值*样本数加起来作为第i个可训练变量的总值 
                # print("len(v)", len(v))                     #还是784和26,26可以理解为模型的参数个数，对应权重值，但是784明显是一张图片，图片*样本数/总的样本数有意义吗
                # print("i: ", i)                             #0和1

        # print("base: ",base)                                #base里面也放了两个array
        # print("len(base[0]): {}, len(base[1]): {}".format(len(base[0]), len(base[1])))       #len(base[0])=784 28*28，一张展开的图片 , len(base[1])=26类别
        averaged_soln = [v / total_weight for v in base]    #[每个可训练变量的总值/总样本数]    可训练变量的值*客户端c样本数/总样本数--最终的权重还是加权平均

        return averaged_soln

    #q-ffl uses aggregate2() <--> fedavg, fedsae use aggregate() 
    #qffl的更新，w^(t+1) = w^(t) - sum(Delta_k^t)/sum(h_k^t)
    def aggregate2(self, weights_before, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        num_clients = len(Deltas)
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]

        return new_solutions

    #为每个客户端生成本轮可以完成的epoch数目，另外，每个客户端对应的epoch要放到client中
    #只要随机数种子一样每次产生的随机数就会一样，那么只要每一轮的随机数种子相同那么就会产生相同的随机数
    def client_epoch(self, c_indices, seed):
      """
      create a normal distribution for every client and return a list of epochs that clients can achive at this round
      Args:
          c_indices: the  indices of all the selected clients
          seed: random seed
      
      Return:
          list of epochs that clients can achieve at this round
      
      """
      np.random.seed(seed)
      num_clients = len(self.clients)
      all_cepoch = self.normal_cepoch(num_clients, seed)
    #   all_cepoch = self.fixed_normal_cepoch(num_clients, round)
      # all_cepoch = self.poisson_cepoch(num_clients, seed)
      
      
      #将每一轮产生的epoch数设置到client中
      for i in range(num_clients):
          self.clients[i].set_ideal_epoch(all_cepoch[i])

      select_cepoch = np.asarray(all_cepoch)[c_indices]      #c_indices随机选中的客户端的下标
      
      return select_cepoch
    
    

    def initial_actual_epoch(self, clients, initial_epoch):
        """       
        Args:
            clients: all the client objects
            initial_epoch: the initital number of actual epoch to evry client
        
        Return:
        
        """
        #初始化actual_epoch
        for c in clients:
            c.set_actual_epoch(initial_epoch)
    
    #原始的产生正态分布epoch的函数
    def normal_cepoch(self, num_clients, seed):
        """
        create a normal distribution for every client and return a list of epochs that clients can achive at this round
        Args:
            c_indices: the  indices of all the selected clients
            seed: random seed
        
        Return:
            list of epochs that clients can achieve at this round
    
        """
        np.random.seed(0)                                                                        
        #num_clients = len(self.clients)                                                            #使用随机数种子0控制每次客户端的μ都是一样的
        #若均值服从均匀分布，有的client  sigma=0
        #cmiu = [random.normalvariate(5.0, 1.0) for i in range(num_clients)]                       #首先为每个客户端选择一个μ
        cmiu = [np.random.uniform(5, 10) for i in range(num_clients)]                            #为每个客户端生成均值miu，μ暂定[0.1,10)
        csigma = [np.random.uniform(cmiu[i]/4, cmiu[i]/2) for i in range(num_clients)]

        all_cepoch = []

        #截断正态分布，截断在（0,μ+2σ）
        #all_cepoch = round(truncnorm.rvs(0, 50, loc=,scale=num_clients, random_state=seed), 1)
        
        #random.seed(seed)
        for i in range(num_clients):
            mu, sigma = cmiu[i], csigma[i]
            lower, upper = 0, 5000  # 截断在[0, +INF]
            val = truncnorm.rvs(lower, upper, loc=mu, scale=sigma, random_state=seed)    #产生正态分布的epoch
            all_cepoch.append(round(val, 1))
            
        #将cmiu,csigma写入文件中，只运行一次,使用write控制
        write = 0                #设置write=0，不写文件
        if(write == 1):
            #将cmiu写入文件中,只运行一次
            cmiu_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/data/cmiu_mnist.txt"
            with open(cmiu_file_name, "w") as f:
                for i in range(num_clients):
                    f.write(str(cmiu[i]) + ' ')
            #将csigma写入文件中，只运行一次
            csigma_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/data/csigma_mnist.txt"
            with open(csigma_file_name, "w") as f:
                for i in range(num_clients):
                    f.write(str(csigma[i]) + ' ')
        return all_cepoch
    
    # 固定每个客户端的epoch，所有客户端的epoch组合起来是一个正态分布
    def fixed_normal_cepoch(self, num_clients, seed):

        np.random.seed(0)
        mu = np.random.uniform(5, 10)
        sigma = mu/4
        # sigma = np.random.uniform(mu/4, mu/2)

        print("in round", seed, "mu:", mu)
        print("sigma:", sigma)

        all_cepoch = []
        for i in range(num_clients):
            lower, upper = 0, 5000;
            val = truncnorm.rvs(lower, upper, loc=mu, scale=sigma, random_state=i)
            all_cepoch.append(round(val, 1))
        print("all_cepoch: ",all_cepoch)
        return all_cepoch

    
    # 泊松分布
    def poisson_cepoch(self, num_clients, seed):
        """generate epoch of clients with poisson distribution

        Args:
            num_clients ([type]): number of clients
            seed ([type]): random seed

        Returns:
            [type]: list of epochs that clients can achieve at this round
        """
        # 每轮客户端的affordable epoch遵循mu为9的泊松分布
        np.random.seed(seed)

        all_cepoch = poisson.rvs(9, size = num_clients)
        print("draw poisson results!\n")
        draw_bar_poisson(all_cepoch, "/home/lab/workspace/lili/TwoTaskCode/FedProx/utils/cepoch_poisson.jpg")

        return all_cepoch


    #客户端有一定概率改变当前epoch正态分布函数
    def ran_normal_cepoch(self, num_clients, seed):
        """
        create a normal distribution for every client and return a list of epochs that clients can achive at this round
        Args:
            c_indices: the  indices of all the selected clients
            seed: random seed
        
        Return:
            list of epochs that clients can achieve at this round
    
        """
        np.random.seed(0)                                                                        
        #num_clients = len(self.clients)                                                            #使用随机数种子0控制每次客户端的μ都是一样的
        #若均值服从均匀分布，有的client  sigma=0
        #首先为每个客户端选择一个μ
        cmiu = [np.random.uniform(5, 10) for i in range(num_clients)]                            #为每个客户端生成均值miu，μ暂定[0.1,10)
        csigma = [np.random.uniform(cmiu[i]/4, cmiu[i]/2) for i in range(num_clients)]

        all_cepoch = []

        #截断正态分布，截断在（0,μ+2σ）
        #all_cepoch = round(truncnorm.rvs(0, 50, loc=,scale=num_clients, random_state=seed), 1)
        
        #random.seed(seed)
        for i in range(num_clients):
            mu, sigma = cmiu[i], csigma[i]
            lower, upper = 0, 5000  # 截断在[0, +INF]
            val = truncnorm.rvs(lower, upper, loc=mu, scale=sigma, random_state=seed)    #产生正态分布的epoch
            all_cepoch.append(round(val, 1))
            
        #将cmiu,csigma写入文件中，只运行一次,使用write控制
        write = 0                #设置write=0，不写文件
        if(write == 1):
            #将cmiu写入文件中,只运行一次
            cmiu_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/data/cmiu_mnist.txt"
            with open(cmiu_file_name, "w") as f:
                for i in range(num_clients):
                    f.write(str(cmiu[i]) + ' ')
            #将csigma写入文件中，只运行一次
            csigma_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/data/csigma_mnist.txt"
            with open(csigma_file_name, "w") as f:
                for i in range(num_clients):
                    f.write(str(csigma[i]) + ' ')
        return all_cepoch

    def beta_cepoch(self, num_clients, seed):
        all_cepoch = []

        np.random.seed(seed)
        numargs = beta.numargs 
        [a, b] = [0.6, ] * numargs
        a, b =4, 6
        #产生β分布的数
        for i in range(num_clients):
            val = round(beta.rvs(a, b, scale = 10), 1)
            while( val <= 0):
                val = round(beta.rvs(a, b, scale = 10), 1)
            all_cepoch.append(val)

        return all_cepoch

    def uniform_cepoch(self, num_clients, seed):
        all_cepoch = []
        np.random.seed(seed)

        all_cepoch = [round(np.random.uniform(0.1, 10), 1) for i in range(num_clients)]
        #print("uniform_cepoch: ", all_cepoch)

        return all_cepoch


    def write_data_to_file(self, trainer, str_dataset, distribution, file_name, test_acc, train_acc, loss, drop_out_percentage, selected_times, test_acc_variance, train_acc_variance):
        '''
        write the training result into files respectively
        '''
        #/home/lab/workspace/lili/MyEditCode/FedProx/result/
        test_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc.txt"
        with open(test_acc_file_name, 'w', encoding='utf-8') as test_f:
            for test_v in test_acc:
                test_f.write(str(test_v) + ' ')

        train_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc.txt"
        with open(train_acc_file_name, 'w', encoding='utf-8') as train_f:
            for train_v in train_acc:
                train_f.write(str(train_v) + ' ')

        loss_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_loss.txt"
        with open(loss_file_name, 'w', encoding='utf-8') as loss_f:
            for loss_v in loss:
                loss_f.write(str(loss_v) + ' ')

        dop_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_dop.txt"
        with open(dop_file_name, 'w', encoding='utf-8') as dop_f:
            for dop_v in drop_out_percentage:
                dop_f.write(str(dop_v) + ' ')

        stimes_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_selected_times.txt"
        with open(stimes_file_name, 'w', encoding='utf-8') as stimes_f:
            for st in selected_times:
                stimes_f.write(str(st) + ' ')

        test_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc_variance.txt"
        with open(test_acc_variance_file_name, 'w', encoding='utf-8') as test_f:
            for test_v in test_acc_variance:
                test_f.write(str(test_v) + ' ')

        train_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc_variance.txt"
        with open(train_acc_variance_file_name, 'w', encoding='utf-8') as train_f:
            for train_v in train_acc_variance:
                train_f.write(str(train_v) + ' ')

    def get_result(self, trainer, str_dataset, distribution, file_name):
        test_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc.txt"
        test_acc_file = open(test_acc_file_name, 'r')
        train_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc.txt"
        train_acc_file = open(train_acc_file_name, 'r')
        loss_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_loss.txt"
        loss_file = open(loss_file_name, 'r')
        dop_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_dop.txt"
        dop_file = open(dop_file_name, 'r')
        stimes_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_selected_times.txt"
        stimes_file = open(stimes_file_name, 'r')
        test_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc_variance.txt"
        test_acc_variance_file = open(test_acc_variance_file_name, 'r')
        train_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc_variance.txt"
        train_acc_variance_file = open(train_acc_variance_file_name, 'r')

        str_test_acc = test_acc_file.read().split()
        test_acc = [float(v) for v in str_test_acc]
       
        str_train_acc = train_acc_file.read().split()
        train_acc = [float(v) for v in str_train_acc]
        
        str_loss = loss_file.read().split()
        loss = [float(v) for v in str_loss]

        str_dop = dop_file.read().split()
        dop = [float(v) for v in str_dop]

        str_stimes = stimes_file.read().split()
        stimes = [float(v) for v in str_stimes]

        str_test_acc_variance = test_acc_variance_file.read().split()
        test_acc_variance = [float(v) for v in str_test_acc_variance]
       
        str_train_acc_variance = train_acc_variance_file.read().split()
        train_acc_variance = [float(v) for v in str_train_acc_variance]
        
        test_acc_file.close()
        train_acc_file.close()
        loss_file.close()
        dop_file.close()
        stimes_file.close()
        test_acc_variance_file.close()
        train_acc_variance_file.close()
        
        print("test_acc: {}\n train_acc: {}\n loss: {}\n dop: {}\n selected times: {}\n test accuracy variance: {}\n train accuracy variance: {}".format(test_acc, train_acc, loss, dop, stimes, test_acc_variance, train_acc_variance))

        return test_acc, train_acc, loss, dop, stimes, test_acc_variance, train_acc_variance

    #用于fedavg，fedavg中没有写选中次数
    def write_data_to_file0(self, trainer, str_dataset, distribution, file_name, test_acc, train_acc, loss, drop_out_percentage, test_acc_variance, train_acc_variance):
        '''
        write the training result into files respectively
        '''
        #/home/lab/workspace/lili/MyEditCode/FedProx/result/
        test_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc.txt"
        with open(test_acc_file_name, 'w', encoding='utf-8') as test_f:
            for test_v in test_acc:
                test_f.write(str(test_v) + ' ')

        train_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc.txt"
        with open(train_acc_file_name, 'w', encoding='utf-8') as train_f:
            for train_v in train_acc:
                train_f.write(str(train_v) + ' ')

        loss_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_loss.txt"
        with open(loss_file_name, 'w', encoding='utf-8') as loss_f:
            for loss_v in loss:
                loss_f.write(str(loss_v) + ' ')

        dop_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_dop.txt"
        with open(dop_file_name, 'w', encoding='utf-8') as dop_f:
            for dop_v in drop_out_percentage:
                dop_f.write(str(dop_v) + ' ')

        test_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc_variance.txt"
        with open(test_acc_variance_file_name, 'w', encoding='utf-8') as test_f:
            for test_v in test_acc_variance:
                test_f.write(str(test_v) + ' ')

        train_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer +"/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc_variance.txt"
        with open(train_acc_variance_file_name, 'w', encoding='utf-8') as train_f:
            for train_v in train_acc_variance:
                train_f.write(str(train_v) + ' ')


    def get_result0(self, trainer, str_dataset, distribution, file_name):
        test_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc.txt"
        test_acc_file = open(test_acc_file_name, 'r')
        train_acc_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc.txt"
        train_acc_file = open(train_acc_file_name, 'r')
        loss_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_loss.txt"
        loss_file = open(loss_file_name, 'r')
        dop_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_dop.txt"
        dop_file = open(dop_file_name, 'r')
        test_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_test_acc_variance.txt"
        test_acc_variance_file = open(test_acc_variance_file_name, 'r')
        train_acc_variance_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/result/" + trainer + "/" + str_dataset + "/" + distribution + "/" + file_name + "_train_acc_variance.txt"
        train_acc_variance_file = open(train_acc_variance_file_name, 'r')

        str_test_acc = test_acc_file.read().split()
        test_acc = [float(v) for v in str_test_acc]

       
        str_train_acc = train_acc_file.read().split()
        train_acc = [float(v) for v in str_train_acc]
        
        str_loss = loss_file.read().split()
        loss = [float(v) for v in str_loss]

        str_dop = dop_file.read().split()
        dop = [float(v) for v in str_dop]

        str_test_acc_variance = test_acc_variance_file.read().split()
        test_acc_variance = [float(v) for v in str_test_acc_variance]
       
        str_train_acc_variance = train_acc_variance_file.read().split()
        train_acc_variance = [float(v) for v in str_train_acc_variance]
        
        test_acc_file.close()
        train_acc_file.close()
        loss_file.close()
        dop_file.close()
        test_acc_variance_file.close()
        train_acc_variance_file.close()
        
        print("test_acc: {}\n train_acc: {}\n loss: {}\n dop: {}\n test accuracy variance: {}\n train accuracy variance: {}".format(test_acc, train_acc, loss, dop, test_acc_variance, train_acc_variance))

        return test_acc, train_acc, loss, dop, test_acc_variance, train_acc_variance
    
    #初始化概率PK的函数
    def get_probability(self):
        #获得基于loss的Vk值并计算概率P
        Vk = []    #存放每个客户端的Vk值
        Pk = []    #存放每个客户端被选择的概率值

        #ids, groups, num_samples, tot_correct, losses = self.train_error_and_loss()
        
        for c in self.clients:
            samples = c.num_samples
            _, c.loss, _ = c.train_error_and_loss()   #此处初始化每个客户端的loss
            loss = c.loss
            vi = 0.01 * loss * math.sqrt(samples)     #0.01为softmax参数
            #707新方案--代码添
            #vi = 1.0/(1+math.exp(c.acc - 1))*vi
            Vk.append(vi)
        #计算Σexp(vi)
        sum_exp = 0
        for vi in Vk:
            sum_exp += math.exp(vi)

        #Pk = [vi/sum(Vk) for vi in Vk]           #直接将Vk值按比例转换成P值
        Pk = [math.exp(vi)/sum_exp for vi in Vk]  #pk=exp(vi)/Σexp(vi)
        self.Pk = Pk
        
        return Pk
    
    #更新被选中客户端的loss
    def update_probability(self):
        Vk = []    #存放每个客户端的Vk值
        Pk = []    #存放每个客户端被选择的概率值

        for c in self.clients:
            samples = c.num_samples
            loss  = c.loss    #这里的loss是交叉熵损失，那么就已经对所有样本求过平均了
            #vi = 0.01 * (loss + self.avg_loss) * math.sqrt(samples)   #每个客户端加上平均loss
            vi = 0.01 * loss * (1.0/math.sqrt(samples))   #每个客户端加上平均loss
            #707新方案--代码添
            #vi = 1.0/(1+math.exp(c.acc - 1))*vi
            Vk.append(vi)
        #计算Σexp(vi),将vi值softmax
        sum_exp = 0
        for vi in Vk:
            sum_exp += math.exp(vi)
        Pk = [math.exp(vi)/sum_exp for vi in Vk]  #pk=exp(vi)/Σexp(vi)   
        
        self.Pk = Pk
    
    #更新传进来的clients的loss   
    def update_loss(self, clients):
        avg_loss = 0
        #将新的loss写到选中的客户端中
        for c in clients:
            print("old loss: ", c.loss)
            _, c.loss, _ = c.train_error_and_loss()
            print("new loss: ", c.loss)
            avg_loss += c.loss
        #更新本轮的平均loss
        avg_loss = avg_loss / len(clients)
        self.avg_loss = avg_loss
        
    # 对客户端进行层次聚类
    def hierarchical_clustering(self, clients_indices, last=True):
        """do hierarchichal clusteting to clients, return random one class of the clustring result
        Args:
            clients_indices ([type]): selected clients at random
        
        Return:
            the indices of a random clustering label
            
        """
        # 将clients_indices转化为clients
        clients = np.array(self.clients)[clients_indices]
        # 聚类接收的参数是整型，所以要把epoch取整
        # 获得客户端上一轮实际运行的epoch
        last_round_epoch = [c.actual_epoch for c in clients]
        # 获得预测的easy epoch, 选择这两个中的任意一个聚类都可
        # 这里predict_epoch必须配合FedSAE使用，否则FedAvg不会预测客户端，那么这个c.two_task[0]将永远不会更新
        predict_epoch = [c.two_task[0] for c in clients]

        #传给fit_predict()的要求是一个array
        truely_epoch = np.array(predict_epoch)
        if(last == False): truely_epoch = np.array(predict_epoch)
        

        # 没有类别参数则默认将聚类两个类别的客户端
        # 计算聚合的合适类别数,这里要好好斟酌
        num_class = int(len(clients) / min(self.clients_per_round, len(clients)))
        #num_class = 2

        clst = cluster.AgglomerativeClustering(num_class)

        clst.fit_predict(truely_epoch.reshape(-1, 1))
        labels = clst.labels_.tolist()

        #整理标签
        dict_clients = {}
        dict_numbers = {}
        dict_epochs = {}
        for i in range(0, len(labels)):
            label = labels[i]
            dict_clients.setdefault(label, []).append(clients_indices[i])   #设置字典的值为list类型存放epoch
            # 规整epoch聚类
            dict_epochs.setdefault(label, []).append(truely_epoch[i])

            if(label not in dict_numbers):
                dict_numbers[label] = labels.count(label)
        print("the result of clustering numbers: ", dict_numbers)
        print("the result of clustering clients: ", dict_clients)
        print("truely_epochs: ", truely_epoch)
        print("the result of clustering epochs: ", dict_epochs)

        # 随便返回一个label类别的客户端, 范围为[0, num_class-1]
        # 一共num_class个类别, 下标从0开始所以最大为num_class-1
        # 要保证返回的这个类别客户端的数目必须大于clients_per_round,使用一个for循环控制
        ran_label = random.randint(0, num_class-1)
        while(len(dict_clients[ran_label]) < self.clients_per_round):
            print("client numbers of this label is less than clients_per_round: ", len(dict_clients[ran_label]))
            ran_label = random.randint(0, num_class-1)
        
        return dict_clients[ran_label]

        

