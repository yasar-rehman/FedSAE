import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])        #fedavg的inner_opt是GD，FedProx是自己定义的一个优化器
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

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
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model)                 #为什么self.latest_model使用在定义之前

            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model)

                total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)

                # solve minimization locally
                #修改fedprox，活跃的运行所有的ideal_epoch，不活跃的运行（0.1，ideal_epoch）个epoch
                if c in active_clients:
                    # soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                    iters = int((c.ideal_epoch - int(c.ideal_epoch)) * total_iters)
                    if(iters > 0):
                        soln, stats = c.solve_iters(num_iters=iters, batch_size=self.batch_size)
                    if(int(c.ideal_epoch) > 0):
                        soln, stats = c.solve_inner(num_epochs=int(c.ideal_epoch), batch_size=self.batch_size)
                    print("train {} epochs".format(c.ideal_epoch))
                else:
                    #视为drop out的客户端只计算其中的[1,E]个epoch，修改的地方
                    #soln, stats = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size)
                    random_epoch = np.random.uniform(0.1, self.num_epochs)

                    iters = int((random_epoch - int(random_epoch)) * total_iters)
                    if(iters > 0):
                        soln, stats = c.solve_iters(num_iters=iters, batch_size=self.batch_size)
                    if(int(random_epoch) > 0):
                        soln, stats = c.solve_inner(num_epochs=int(random_epoch), batch_size=self.batch_size)
                    print("train {} epochs".format(random_epoch))

                # gather solutions from client
                csolns.append(soln)
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
