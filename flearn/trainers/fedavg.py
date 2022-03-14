import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad

#fedavg也是服务器使用的优化器，获得参数训练客户端
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])      #inner_opt即梯度下降算法
        super(Server, self).__init__(params, learner, dataset)                           #显式调用父类的构造函数，因为子类重写了__init__函数所以不会自动调用父类的__init__函数,和java一样

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))             #clients_per_round从哪里获得的,只有run_fedavg.sh中有一个clients_per_round，为什么这里可以用
        
        
        acc = -1
        drop_out_percentage = []
        test_acc = []
        train_acc = []
        loss = []

        #test_acc_file, train_acc_file, loss_file = self.makefile("mnist")

        for i in range(self.num_rounds):                                                 #num_rounds也是传进来的参数
            # test model
            if i % self.eval_every == 0:   
                #eval_every传进来的参数，每几轮测试一下
                _stats = self.test()  # have set the latest model for all clients         #stats是ids, groups, num_samples, tot_correct，全是数组
                #父类的train_error_and_loss调用Client类的train_error_and_loss，该函数又调用mclr等模型中的test方法真正开始训练
                _stats_train = self.train_error_and_loss()                                #stats_train的形式ids, groups, num_samples, tot_correct, losses

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(_stats_train[3]) * 1.0 / np.sum(_stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(_stats_train[4], _stats_train[2]) * 1.0 / np.sum(_stats_train[2])))   #np.dot()点乘两个列表，对应相乘再相加，如果是两个矩阵就是矩阵乘法
                v_test_acc = np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])
                v_train_acc = np.sum(_stats_train[3]) * 1.0 / np.sum(_stats_train[2])
                v_loss = np.dot(_stats_train[4], _stats_train[2]) * 1.0 / np.sum(_stats_train[2])
                test_acc.append(v_test_acc)
                train_acc.append(v_train_acc)
                loss.append(v_loss)

            #self.select_clients self.hierarchical_select_clients
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling，随机选客户端，indices是选中的下标，selected_clients是选中的客户端
            print('debug10:')
            np.random.seed(i)                                  #设置随机数种子，使用了相同的种子则会产生同样的随机数，每个数据集产生的客户端随机数相同
            #active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)   #从选中的客户端中再选择一部分作为活跃客户端完成所有epoch，剩下的只完成一部分，视为drop out，active_clients是实例化的客户端
            
            active_clients = []
            #active_num = 0                             #记录本轮没有drop out的client的个数
            
            #此时的fedavg变为如果分配的epoch大于ideal_epoch则无法完成,fedavg无法自适应变化epoch所以每一轮都是actual_epoch
            actual_epoch = self.num_epochs
            #actual_epoch = actual_epoch + i * 0.05
            for s_c in selected_clients:
                if(s_c.get_ideal_epoch() > actual_epoch):
                    active_clients.append(s_c)

            csolns = []                                        # buffer for receiving client solutions

            #for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
            for c in active_clients:
                
                # communicate the latest model
                c.set_params(self.latest_model)                #每个客户端在开始计算之前都将模型参数置为latest_model，因为客户端不是同时多个线程一起计算的，所以一个客户端计算完了之后要复位为分发模型的状态

                # solve minimization locally                   #修改的时候要改成每个客户端根据自身情况使用策略猜测能完成的epoch数
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)   #soln, stats=(self.num_samples, soln), (bytes_w, comp, bytes_r)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                print("stats: {}".format(stats))
                print("c.ideal_epoch: ", c.ideal_epoch)
                self.metrics.update(rnd=i, cid=c.id, stats=stats)   #metrics是model_utils中Metrics类的实例


            dop = 1.0 - len(active_clients) / len(selected_clients)
            drop_out_percentage.append(dop)
            print(drop_out_percentage)
            print('At round {} drop out percentage: {}'.format(i+1, dop))
            
            # update models
            if(len(csolns) == 0):
                continue
            
            self.latest_model = self.aggregate(csolns)              #聚合客户端的梯度
            
            if(np.sum(_stats[3]) * 1.0 / np.sum(_stats[2]) > acc):
                acc = np.sum(_stats[3]) * 1.0 / np.sum(_stats[2])
            print("acc: ", acc)

        print("highest accuracy: ", acc)
        print("lowest drop out percentage: ", min(drop_out_percentage))
        print("average drop out percentage: ", sum(drop_out_percentage) / len(drop_out_percentage))
        
        
        # final test model
        stats = self.test()                                         #stats是ids, groups, num_samples, tot_correct，全是数组
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)

        if(np.sum(stats[3]) * 1.0 / np.sum(stats[2]) > acc):        #最后再计算一次测试集精度
            acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        
        
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        
        v_test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        v_train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        test_acc.append(v_test_acc)
        train_acc.append(v_train_acc)

        self.write_data_to_file0(trainer="fedavg", str_dataset="mnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="mnist_fedavg_hc_num_epoch_15_c30_200r_mclr", test_acc=test_acc, train_acc=train_acc, loss=loss, drop_out_percentage = drop_out_percentage)
        self.get_result0(trainer="fedavg", str_dataset="mnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="mnist_fedavg_hc_num_epoch_15_c30_200r_mclr")



