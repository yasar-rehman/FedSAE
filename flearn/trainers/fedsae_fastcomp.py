import numpy as np
import random
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
INITIAL_EPOCH = 1

#有阈值，快增慢增任务
#复制的FedProx类，用来改写成客户端自适应调整epoch
#服务器分发给客户端easy和difficult的任务，客户端并依据任务猜测区间
##快启动fast的对比试验，drop out之后从0开始增
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        print(params)
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])        #fedavg的inner_opt是GD，FedProx是自己定义的一个优化器
        super(Server, self).__init__(params, learner, dataset)

    def deal_interval(self, c, easy, difficult, epoch_step, drop_num):
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

      #调整下一轮的easy和difficult
      if(len(c.theta) >= 2):                             #满足阈值使用条件,θ的值必须足够多，EMA取2，MA取1
        theta = c.theta[-1]
        delta_1 = 3.0
        delta_2 = 3.0
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
            
        if(easy <= c.ideal_epoch and c.ideal_epoch < difficult):      #client只能完成Easy难度的任务
            c.set_two_task_achieved(1, 0)
            actual_epoch = easy                        #client能完成的epoch数介于Easy和difficult之间，那么实际能完成的任务只有小的任务
            
            if(theta <= easy):
              easy = min(easy+delta_2, difficult-3.0)
              difficult = max(easy+delta_2, difficult-3.0)
            if(easy < theta and theta <= difficult):
              easy = min(easy+delta_1, difficult-3.0)
              difficult = max(easy+delta_1, difficult-3.0)
            if(theta > difficult):
              easy = min(easy+delta_1, difficult-3.0)
              difficult =max(easy+delta_1, difficult-3.0)
            
            bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()

        if(easy > c.ideal_epoch):                                   #client无法完成任务
            print("train 0 epochs")
            isTrain = 0                           #标记并未参与训练，提示后面的csolns不需要再次加soln到列表里面
            drop_num += 1
            c.set_two_task_achieved(0, 0)
            actual_epoch = 0

            easy -= 3.0               #drop out之后任务变成初始值
            difficult -= 3.0
            # easy /= 2.0
            # difficult /= 2.0

            bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()
      else:                                              #不满足阈值使用条件，当bit_epoch的历史值不足2个时，使用最原始的bit_epoch方案
        increament = 1.0
        if(c.ideal_epoch >= difficult):                  #client能完成difficult的任务
            c.set_two_task_achieved(1, 1)
            actual_epoch = difficult

            if(c.bit_epoch > 0):
              easy = min(c.bit_epoch, difficult+increament)
              difficult = max(c.bit_epoch, difficult+increament)
            
            bit_epoch = int(difficult/epoch_step) * epoch_step
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()
            
        if(easy <= c.ideal_epoch and c.ideal_epoch < difficult):      #client只能完成Easy难度的任务
            c.set_two_task_achieved(1, 0)
            actual_epoch = easy                        #client能完成的epoch数介于Easy和difficult之间，那么实际能完成的任务只有小的任务
            
            if(c.bit_epoch > 0):
              easy = min(c.bit_epoch, easy+increament, difficult-3.0)
              difficult = max(c.bit_epoch, easy+increament, difficult-3.0)
              # easy = min(c.bit_epoch, easy+increament, difficult/2.0)
              # difficult = max(c.bit_epoch, easy+increament, difficult/2.0)
            
            bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()

        if(easy > c.ideal_epoch):                                   #client无法完成任务
            print("train 0 epochs")
            isTrain = 0                           #标记并未参与训练，提示后面的csolns不需要再次加soln到列表里面
            drop_num += 1
            c.set_two_task_achieved(0, 0)
            actual_epoch = 0

            # easy = 1.0
            # difficult = 2.0
            if(c.bit_epoch > 0):
              easy = min(easy-3.0, c.bit_epoch)
              difficult = max(easy-3.0, c.bit_epoch)
              # easy = min(easy/2.0, c.bit_epoch)
              # difficult = max(easy/2.0, c.bit_epoch)

            bit_epoch = int(c.ideal_epoch/epoch_step) * epoch_step             #更新下一轮的bit_epoch
            c.add_Elist(bit_epoch)
            r0, r1 = c.update_tr()
            
      return easy, difficult, actual_epoch, bit_epoch, drop_num

    def train(self):
        '''Train using Federated epoch'''
        print('Training with {} workers ---'.format(self.clients_per_round))
 
        acc = -1                                #标记最高精度
        drop_out_percentage = []   #buffer for recerving client drop out percentages
        test_acc = []
        train_acc = []
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

                v_test_acc = np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])
                v_train_acc = np.sum(_stats_train[3]) * 1.0 / np.sum(_stats_train[2])
                v_loss = np.dot(_stats_train[4], _stats_train[2]) * 1.0 / np.sum(_stats_train[2])
                
                test_acc.append(v_test_acc)
                train_acc.append(v_train_acc)
                loss.append(v_loss)

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

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model)                 #为什么self.latest_model使用在定义之前

            drop_num = 0                             #记录本轮没有drop out的client的个数
            epoch_step = 1.0                         #初始化epoch_step即client刷新该位的频率，每隔epoch_step个epoch更新一次
            increament = 1.0                         #调整区间时每次增加的epoch增量
            
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
                
                #total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)
                total_iters = int(c.num_samples / self.batch_size)
                
                # solve minimization locally
                #这里改写为自适应分配epoch数目,根据上一次的运行情况来调整本次的epoch数目
                #第一次运行的时候调整achieved
                               
                print("selected_times:{}".format(c.selected_times))
                isTrain = 1
                easy = c.two_task[0]
                difficult = c.two_task[1]
                
                #根据历史进行任务预测
                easy, difficult, actual_epoch, bit_epoch, drop_num = self.deal_interval(c, easy, difficult, epoch_step, drop_num)    
                print("debug1: ideal_epoch: {}, bit_epoch: {}, [easy, difficult]: [{}, {}]".format(c.ideal_epoch, bit_epoch, easy, difficult))
                print("debug1: before judge, actual_epoch: ", actual_epoch)

                c.set_bit_epoch(bit_epoch)

                #更新下一轮的开始任务区间为本轮使用的任务区间
                c.set_two_task([easy, difficult])
                
                if(actual_epoch > MAX_EPOCH): actual_epoch = MAX_EPOCH                   #判断actual_epoch是否超出了最大的epoch值

                #进行训练
                iters = int((actual_epoch - int(actual_epoch)) * total_iters)             #计算epoch小数部分，转换成iteration,然后计算epoch的整数部分
                if(iters > 0):
                    soln, stats = c.solve_iters(num_iters=iters, batch_size=self.batch_size)  #stats只有计算了才会覆盖掉上一次计算的，从而更新
                if (int(actual_epoch) > 0):                                                 #计算epoch的整数部分
                    soln, stats = c.solve_inner(num_epochs=int(actual_epoch), batch_size=self.batch_size)
                
                c.set_t_actual_epoch(actual_epoch)                                         #这一步不知道有什么用，先写着
                
                print("debug1: train {} epochs".format(actual_epoch))
                
                #更新模型，聚合等操作
                # update the selected times of client c
                c.set_selected_times(c.selected_times+1) 
                
                # gather solutions from client
                if(isTrain == 0):     #isTrain的作用是防止当前客户端训练的epoch数为0，soln没有更新所以append了前一个客户端的soln，只有训练了才append
                    continue
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
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

            
            if(np.sum(_stats[3]) * 1.0 / np.sum(_stats[2]) > acc):
                acc = np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])
            print("acc: ", acc)
        
        print("highest accuracy: ", acc)
        print("lowest drop out percentage: ", min(drop_out_percentage))
        print("average drop out percentage: ", sum(drop_out_percentage)/len(drop_out_percentage))
        
        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)

        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))

        v_test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        v_train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        test_acc.append(v_test_acc)
        train_acc.append(v_train_acc)

        self.write_data_to_file(trainer="fedsae", str_dataset="mnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="fastcomp5_EMA_δ13_δ23_drop3_α0.95", test_acc=test_acc, train_acc=train_acc, loss=loss, drop_out_percentage = drop_out_percentage)
        self.get_result(trainer="fedsae", str_dataset="mnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="fastcomp5_EMA_δ13_δ23_drop3_α0.95")



        #打印出猜测的区间two_task和μ的区别
        #首先读μ
        hit = 0.0
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
        print("the percentage of hit: ", hit/client_range)

