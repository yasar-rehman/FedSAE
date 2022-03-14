import numpy as np
import random
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
INITIAL_EPOCH = 1

#复制的FedProx类，用来改写成客户端自适应调整epoch
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        print(params)
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])        #fedavg的inner_opt是GD，FedProx是自己定义的一个优化器
        super(Server, self).__init__(params, learner, dataset)

    
    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

   
        acc = -1                                #标记最高精度
        drop_out_percentage = []   #buffer for recerving client drop out percentages
        test_acc = []
        test_acc_variance = []    #每轮客户端的测试精度variance
        train_acc = []
        train_acc_variance = []   #每轮客户端的训练精度variance
        loss = []

        for i in range(self.num_rounds):
            
            # test model
            if i % self.eval_every == 0:
                _stats = self.test() # have set the latest model for all clients
                _stats_train = self.train_error_and_loss()  #此处的stats改变了stats的值，后面如果运行的是0个epoch就会产生冲突的stats

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(_stats[3])*1.0/np.sum(_stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(_stats_train[3])*1.0/np.sum(_stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(_stats_train[4], _stats_train[2])*1.0/np.sum(_stats_train[2])))

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
            

            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model)
                

                #total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)
                total_iters = int(c.num_samples / self.batch_size)
                
                # solve minimization locally
                #这里改写为自适应分配epoch数目,根据上一次的运行情况来调整本次的epoch数目
                #第一次运行的时候调整achieved
                
                #测试，每个客户端运行全部的ideal_epoch
                # if(iters > 0):
                #     soln, stats = c.solve_iters(num_iters=iters, batch_size=self.batch_size)  #stats只有计算了才会覆盖掉上一次计算的，从而更新
                # if (int(c.ideal_epoch) > 0):                                                 #计算epoch的整数部分
                #     soln, stats = c.solve_inner(num_epochs=int(c.ideal_epoch), batch_size=self.batch_size)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                print("train {} epochs".format(c.ideal_epoch))
                isTrain = 1
                        
                
                # gather solutions from client
                if(isTrain == 1):     #isTrain的作用是防止当前客户端训练的epoch数为0，soln没有更新所以append了前一个客户端的soln，只有训练了才append
                    csolns.append(soln)
            
                # track communication cost  #此处的问题是如果第一轮所有的客户端的ideal_epoch都小于了actual_epoch那么将不会产生任何soln和stats，那么就会报错
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

            #print("drop_num: {}".format(drop_num))
            dop = drop_num / len(selected_clients)
            drop_out_percentage.append(dop)
            print(drop_out_percentage)
            print('At round {} drop out percentage: {}'.format(i+1, dop))

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
        v_test_acc_variance = np.std(stats[4], ddof=1)
        v_train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        v_train_acc_variance = np.std(stats_train[5], ddof=1)
        test_acc.append(v_test_acc)
        test_acc_variance.append(v_test_acc_variance)
        train_acc.append(v_train_acc)
        train_acc_variance.append(v_train_acc_variance)


        self.write_data_to_file0(trainer="fedNDrop", str_dataset="fmnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="fmnist_fullworload_10c_300r_mclr", test_acc=test_acc, train_acc=train_acc, loss=loss, drop_out_percentage = drop_out_percentage, test_acc_variance=test_acc_variance, train_acc_variance=train_acc_variance)
        self.get_result0(trainer="fedNDrop", str_dataset="fmnist", distribution="normal_σ=ran(0.25μ_0.5μ)", file_name="fmnist_fullworload_10c_300r_mclr")


