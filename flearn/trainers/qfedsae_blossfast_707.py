import numpy as np
import random
import math
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
INITIAL_EPOCH = 1


#目标函数为q-FFL的fast
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using FedSAE blossfast to Train')
        print(params)
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])        #fedavg的inner_opt是GD，FedProx是自己定义的一个优化器
        super(Server, self).__init__(params, learner, dataset)
        self.q = 0.1

    def deal_interval(self, c, easy, difficult, epoch_step, drop_num, loss_threshold):
      '''
      c：客户端
      easy，difficult：本轮服务器分配给客户端的两个任务
      epoch_step：进行最近epoch预测时设定的通信频率，每计算1个epoch就更新一次bit_epoch位还是更多
      '''
      #EMA改进版递归计算theta θ(t)=α*θ(t-1) + (1-α)*E(t-1)
      def CalEMATheta(t):
        a = 0.95  #0.716531
        if(t == 1):
          return c.Elist[0]
        else:
          return a*CalEMATheta(t-1)+(1-a)*c.Elist[t-1]
      
      #MA
      def CalMATheta():
        return sum(c.Elist)/len(c.Elist)

      if(len(c.Elist) >= 2):   #CalEMATheta(t)和>=2搭配
        t = len(c.Elist)
        theta = CalEMATheta(t)
        #theta = CalMATheta()
        c.add_theta(theta)
        print("theta:", theta)
      
      #增加基于loss调整[easy, difficult]区间的AFL部分
      #_tot_correct, _loss, _num_samples = c.train_error_and_loss()           #获得本客户端的loss
      _loss = c.loss
      upload_rate = 0                                                      #upload_rate控制短点上传间隔的epoch数目
      magnify = 1.0                                                          #magnify控制扩大的区间
      segment = 0                                                            #分段上传隔的epoch
      upload_crash = []                                                      #列表保存遇到epoch为哪些值的倍数时需要上传
      if _loss > loss_threshold:
        easy = easy * magnify
        difficult = difficult * magnify
        segment = math.floor(difficult * upload_rate)                                    #分段上传的segment
        upload_crash.append(segment)

      #调整下一轮的easy和difficult
      if(len(c.theta) >= 2):                             #满足阈值使用条件,θ的值必须足够多，EMA取2，MA取1
        theta = c.theta[-1]
        delta_1 = 3.0
        delta_2 = 1.0
        if(c.ideal_epoch >= difficult):                  #client能完成difficult的任务
          c.set_two_task_achieved(1, 1)
          actual_epoch = difficult
          
          if(theta <= easy):
            easy = min(easy+delta_2, difficult+delta_2)
            difficult = max(easy+delta_2, difficult+delta_2)
          if(easy < theta and theta <= difficult):
            easy = min(easy+delta_1, difficult+delta_2)
            difficult = max(easy+delta_1, difficult+delta_2)
          if(theta > difficult):
            easy = min(easy+delta_1, difficult+delta_1)
            difficult = max(easy+delta_1, difficult+delta_1)
          
          bit_epoch = int(difficult/epoch_step) * epoch_step
          c.add_Elist(bit_epoch)
          r0, r1 = c.update_tr()

          if(segment > 0):
            upload_crash.append(easy)                            #此时遇到easy和difficult都需要上传
            upload_crash.append(difficult)
            
        if(easy <= c.ideal_epoch and c.ideal_epoch < difficult):      #client只能完成Easy难度的任务
            c.set_two_task_achieved(1, 0)
            actual_epoch = easy                        #client能完成的epoch数介于Easy和difficult之间，那么实际能完成的任务只有小的任务
            
            if(theta <= easy):
              easy = min(easy+delta_2, difficult/2.0)
              difficult = max(easy+delta_2, difficult/2.0)
            if(easy < theta and theta <= difficult):
              easy = min(easy+delta_1, difficult/2.0)
              difficult = max(easy+delta_1, difficult/2.0)
            if(theta > difficult):
              easy = min(easy+delta_1, difficult/2.0)
              difficult =max(easy+delta_1, difficult/2.0)
            
            bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()

            if(segment > 0 and c.ideal_epoch-easy > segment):
              actual_epoch = int(c.ideal_epoch/segment) * segment
              upload_crash.append(easy) 

        if(easy > c.ideal_epoch):                                   #client无法完成任务
            print("train 0 epochs")
            isTrain = 0                           #标记并未参与训练，提示后面的csolns不需要再次加soln到列表里面
            drop_num += 1
            c.set_two_task_achieved(0, 0)
            actual_epoch = 0

            easy /= 2.0
            difficult /= 2.0

            bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()

            if(segment > 0 and c.ideal_epoch > segment):                       #判断能否完成一个segment，如果能则是可以运行一个segment的
              actual_epoch = int(c.ideal_epoch/segment) * segment
      else:                                              #不满足阈值使用条件，当bit_epoch的历史值不足2个时，使用最原始的bit_epoch方案
        increament = 1.0
        if(c.ideal_epoch >= difficult):                  
          c.set_two_task_achieved(1, 1)#client能完成difficult的任务
          actual_epoch = difficult

          if(c.bit_epoch > 0):
            easy = min(c.bit_epoch, difficult+increament)
            difficult = max(c.bit_epoch, difficult+increament)
          
          bit_epoch = int(difficult/epoch_step) * epoch_step
          c.add_Elist(bit_epoch)
          r0, r1 = c.update_tr()

          if(segment>0):
            upload_crash.append(easy)                            #此时遇到easy和difficult都需要上传
            upload_crash.append(difficult)
            
        if(easy <= c.ideal_epoch and c.ideal_epoch < difficult):      #client只能完成Easy难度的任务
          c.set_two_task_achieved(1, 0)
          actual_epoch = easy                        #client能完成的epoch数介于Easy和difficult之间，那么实际能完成的任务只有小的任务
          
          if(c.bit_epoch > 0):
            easy = min(c.bit_epoch, easy+increament, difficult/2.0)
            difficult = max(c.bit_epoch, easy+increament, difficult/2.0)
          
          bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
          c.add_Elist(bit_epoch)
          r0, r1 = c.update_tr()

          if(segment > 0 and c.ideal_epoch-easy > segment):
            actual_epoch = int(c.ideal_epoch/segment) * segment
            upload_crash.append(easy)                                        #因为能力比easy大了一个或者多个segment，此时遇到easy也需要上传


        if(easy > c.ideal_epoch):                                   #client无法完成任务
          print("train 0 epochs")
          isTrain = 0                           #标记并未参与训练，提示后面的csolns不需要再次加soln到列表里面
          drop_num += 1
          c.set_two_task_achieved(0, 0)
          actual_epoch = 0

          if(c.bit_epoch > 0):
            easy = min(easy/2.0, c.bit_epoch)
            difficult = max(easy/2.0, c.bit_epoch)

          bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
          c.add_Elist(bit_epoch)
          r0, r1 = c.update_tr()

          if(segment > 0 and c.ideal_epoch > segment):                       #判断能否完成一个segment，如果能则是可以运行一个segment的
            actual_epoch = int(c.ideal_epoch/segment) * segment
            
      return easy, difficult, actual_epoch, bit_epoch, drop_num, segment, upload_crash

    def train(self):
        '''Train using Federated epoch'''
        print('Training with {} workers ---'.format(self.clients_per_round))
 
        acc = -1                                #标记最高精度
        acc_variance = float('inf')                       #标记最高精度方差
        drop_out_percentage = []   #buffer for recerving client drop out percentages
        test_acc = []
        test_acc_variance = []    #每轮客户端的测试精度variance
        train_acc = []
        train_acc_variance = [] 
        loss = []
        MAX_EPOCH = 15.0         #MAX_EPOCH控制客户端能运行的最大epoch数目，即使客户端能运行超过MAX_EPOCH也只使它完成MAX_EPOCH的任务，服务器分配任务的时候必须下一道禁制
        for i in range(self.num_rounds):
            
            # test model
            if i % self.eval_every == 0:
                _stats = self.test() # have set the latest model for all clients
                _stats_train = self.train_error_and_loss()  #此处的stats改变了stats的值，后面如果运行的是0个epoch就会产生冲突的stats

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(_stats[3])*1.0/np.sum(_stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(_stats_train[3])*1.0/np.sum(_stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(_stats_train[4], _stats_train[2])*1.0/np.sum(_stats_train[2])))
                tqdm.write('At round {} training accuracy variance: {}'.format(i, np.std(_stats_train[5], ddof=1)))

                v_test_acc = np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])
                #计算每轮客户端的测试精度variance
                v_test_acc_variance = np.std(_stats[4], ddof=1)
                v_train_acc = np.sum(_stats_train[3]) * 1.0 / np.sum(_stats_train[2])
                #计算每轮客户端的训练精度variance
                v_train_acc_variance = np.std(_stats_train[5], ddof=1)
                v_loss = np.dot(_stats_train[4], _stats_train[2]) * 1.0 / np.sum(_stats_train[2])
                
                test_acc.append(v_test_acc)
                test_acc_variance.append(v_test_acc_variance)
                train_acc.append(v_train_acc)
                train_acc_variance.append(v_train_acc_variance)
                loss.append(v_loss)

            #if(i>0): self.update_probability(indices)                                        #除了第一轮，其余每轮均在选客户端之前更新被选概率
            model_len = process_grad(self.latest_model).size                                 #process_grad()将模型展平可以看到模型有多少个梯度参数
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)                                   #Client的get_grads()调用了Model的get_grads()
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0                                                                  #计算局部梯度和全局梯度的方差
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            tqdm.write('gradient difference: {}'.format(difference))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            #active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)
            active_clients = selected_clients
            Deltas =[]
            hs = []

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model)                 #为什么self.latest_model使用在定义之前

            drop_num = 0                             #记录本轮没有drop out的client的个数
            epoch_step = 1.0                         #初始化epoch_step即client刷新该位的频率，每隔epoch_step个epoch更新一次
            increament = 1.0                         #调整区间时每次增加的epoch增量
            loss_threshold = v_loss                  #定义loss的阈值为上一轮的加权平均

            for idx, c in enumerate(selected_clients.tolist()):
                '''
                #  横向
                # [easy, difficult]
                #    √       √            [easy, difficult + r1]
                #    √       ×            [easy, difficult / 2.0]
                #    ×       ×            [easy/2, easy/2 + 1]
                # 纵向使用最近两次的完成情况更新

                # 使用变化的增量(highest_acc=0.751039501,drop=0.0945,hit=0.575)
                # [easy, difficult]
                #    √       √            [easy + r0, difficult + r1] | [easy, difficult+r1] | [easy, difficult]
                #    √       ×            [easy, difficult / 2.0] | [easy, difficult/2]
                #    ×       ×            [easy/2, easy/2 + 1] | [easy/2, easy/2 + 1]

                #client增设bit位存放能完成的epoch的近似值，有多近似将取决于步长(highest_acc=0.7396048,drop=0.0955,hit=0.575)
                # [easy, difficult]
                #    √       √            [easy/2, bit](均选较大的在左，较小的在右)
                #    √       ×            [bit, difficult/2]
                #    ×       ×            [bit, difficult+a]

                #client增设bit位存放停止计算时的epoch的近似值，有多近似将取决于步长(highest_acc=0.753638,drop=0.0945,hit=0.575)
                # *************效果最佳**************************
                # [easy, difficult]       
                #    √       √            [easy/2, bit](均选较小的在左，较大的在右)
                #    √       ×            [bit, easy, difficult/2]
                #    ×       ×            [bit, difficult+a]

                #另一种简单的AIMD算法的变形，增量使用固定增量(highest_acc=0.7463617,drop=0.03,hit=0.2)
                # [easy, difficult]
                #    √       √            [difficult, difficult+a](均选较小的在左，较大的在右)
                #    √       ×            [easy+a, difficlult/2]
                #    ×       ×            [easy/2, difficult/2]
                '''
                # communicate the latest model
                c.set_params(self.latest_model)
                
                total_iters = int(c.num_samples / self.batch_size)
                
                # solve minimization locally
                #这里改写为自适应分配epoch数目,根据上一次的运行情况来调整本次的epoch数目
                #第一次运行的时候调整achieved
                               
                print("selected_times:{}".format(c.selected_times))
                isTrain = 1
                easy = c.two_task[0]
                difficult = c.two_task[1]
                
                #根据历史进行任务预测
                easy, difficult, actual_epoch, bit_epoch, drop_num, segment, upload_crash = self.deal_interval(c, easy, difficult, epoch_step, drop_num, loss_threshold)    
                print("debug1: ideal_epoch: {}, bit_epoch: {}, [easy, difficult]: [{}, {}]".format(c.ideal_epoch, bit_epoch, easy, difficult))
                print("debug1: before judge, actual_epoch: ", actual_epoch)

                c.set_bit_epoch(bit_epoch)

                #更新下一轮的开始任务区间为本轮使用的任务区间
                c.set_two_task([easy, difficult])
                
                if(actual_epoch > MAX_EPOCH): actual_epoch = MAX_EPOCH                   #判断actual_epoch是否超出了最大的epoch值

                #进行训练
                iters = int((actual_epoch - int(actual_epoch)) * total_iters)             #计算epoch小数部分，转换成iteration,然后计算epoch的整数部分
                
                #q-ffl
                weights_before = c.get_params()
                q_loss = c.get_loss()               #compute loss on the whole training data, with respect to the starting point(无)                

                if(segment > 0):
                  #soln, stats = c.solve_entire_segment_upload_keeplast(upload_crash=upload_crash, num_epochs=int(actual_epoch), num_iters=iters, batch_size=self.batch_size, segment=segment)
                  soln, stats = c.solve_entire_segment_upload(upload_crash=upload_crash, num_epochs=int(actual_epoch), num_iters=iters, batch_size=self.batch_size, segment=segment)
                else:
                  soln, stats = c.solve_entire(num_epochs=int(actual_epoch), num_iters=iters, batch_size=self.batch_size)
                #print("debug1: train {} epochs".format(actual_epoch))

                c.set_t_actual_epoch(actual_epoch)                                         #这一步不知道有什么用，先写着
                
                print("debug1: train {} epochs".format(actual_epoch))
                
                #q-ffl
                new_weights = soln[1]
                #plug in the weight updates into the gradient:delta-w_k^t=L(w^t - bar(w)_k^t+1)  
                grads = [(u - v) * 1.0 / self.learning_rate for u, v in zip(weights_before, new_weights)]
                #delta_k^t = F_k^q(w^t)*delta-w_k^t
                Deltas.append([np.float_power(q_loss+1e-10, self.q) * grad for grad in grads])
                # estimation of the local Lipchitz constant
                # h_k^t = q * F_k^(q-1)(w^t) * ||delta-w_k^t||^2 + LF_k^q(w^t)
                hs.append(self.q * np.float_power(q_loss+1e-10, (self.q-1)) * norm_grad(grads) + (1.0/self.learning_rate) * np.float_power(q_loss+1e-10, self.q))
                
                #更新模型，聚合等操作
                # update the selected times of client c
                c.set_selected_times(c.selected_times+1) 
                
                # gather solutions from client
                if(isTrain == 0):     #isTrain的作用是防止当前客户端训练的epoch数为0，soln没有更新所以append了前一个客户端的soln，只有训练了才append
                    continue
                
                if(segment > 0):
                  for s in soln:
                    csolns.append(s)             #extend将soln列表打开并把其中的加到csolns里面  
                else:
                  csolns.append(soln)
                
                # track communication cost  #此处的问题是如果第一轮所有的客户端的ideal_epoch都小于了actual_epoch那么将不会产生任何soln和stats，那么就会报错
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            #print("drop_num: {}".format(drop_num))
            dop = drop_num / len(selected_clients)
            drop_out_percentage.append(dop)
            print(drop_out_percentage)
            print('At round {} drop out percentage: {}'.format(i+1, dop))

            # update models
            if(len(csolns) == 0):
                continue
            #self.latest_model = self.aggregate(csolns)     #普通的聚合
            #q-ffl:aggregate using the dynamix step-size
            self.latest_model = self.aggregate2(weights_before, Deltas, hs)
            self.client_model.set_params(self.latest_model)
            
            #更新被选中客户端的loss,Pk
            self.update_loss([self.clients[id] for id in indices])
            self.update_probability()
            
            if(np.sum(_stats[3]) * 1.0 / np.sum(_stats[2]) > acc):
                acc = np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])
            print("acc: ", acc)

            if(np.std(_stats[4], ddof=1) < acc_variance):
              acc_variance = np.std(_stats[4], ddof=1)
            print("acc_variance: ", acc_variance)
        
        print("highest accuracy: ", acc)
        print("lowest drop out percentage: ", min(drop_out_percentage))
        print("average drop out percentage: ", sum(drop_out_percentage)/len(drop_out_percentage))
        print("lowest accuracy variance: ", acc_variance)
        
        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)

        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))

        v_test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        v_test_acc_variance = np.std(stats[4], ddof=1)
        v_train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        v_train_acc_variance = np.std(stats_train[5], ddof=1)
        test_acc.append(v_test_acc)
        test_acc_variance.append(v_test_acc_variance)
        train_acc.append(v_train_acc)
        train_acc_variance.append(v_train_acc_variance)


        #获得所有客户端的选中次数
        selected_times = []
        for c in self.clients:
          selected_times.append(c.selected_times)

        self.write_data_to_file(trainer="fedsae_correctblossfast", str_dataset="fmnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="fmnist_fast_loss_c10_300r_mclr", test_acc=test_acc, train_acc=train_acc, loss=loss, drop_out_percentage = drop_out_percentage, selected_times=selected_times, test_acc_variance=test_acc_variance, train_acc_variance=train_acc_variance)
        self.get_result(trainer="fedsae_correctblossfast", str_dataset="fmnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="fmnist_fast_loss_c10_300r_mclr")



        #打印出猜测的区间two_task和μ的区别
        #首先读μ
        hit = 0.0
        selected = 0
        cmiu_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/data/cmiu_mnist.txt"
        with open(cmiu_file_name, "r") as f:
            str_cmiu = f.read().split()
        cmiu = [float(v) for v in str_cmiu]

        csigma_file_name = "/home/lab/workspace/lili/TwoTaskCode/FedProx/data/csigma_mnist.txt"
        with open(csigma_file_name, "r") as f:
          str_csigma = f.read().split()
        csigma = [float(v) for v in str_csigma]

        client_range = len(self.clients)           #可选值还有len(self.clients)，用来控制训练的客户端范围,针对客户端数量大的数据集
        for i in range(client_range):
            c = self.clients[i]
            print("[easy, difficult]: [{},{}], miu={}, [miu-sigma, miu+sigma]: [{},{}]".format(c.two_task[0], c.two_task[1], cmiu[i], cmiu[i]-csigma[i], cmiu[i]+csigma[i]))
            print("internal step: [{}, {}]".format(cmiu[i]-csigma[i]-c.two_task[0], cmiu[i]+csigma[i]-c.two_task[1]))
            if(cmiu[i]>c.two_task[0] and cmiu[i]<c.two_task[1]): hit += 1
            if(not(c.two_task[0]== 1.0 and c.two_task[1] == 2.0)): selected += 1
        print("the percentage of hit: ", hit/client_range)
        print("the percentage of selected: ", selected/client_range)

