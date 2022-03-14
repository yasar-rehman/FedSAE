import numpy as np

class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None, ideal_epoch=0.1, 
    actual_epoch=0.2, achieved=None, selected_times=0, r=0.2, two_task=[1, 2], two_task_achieved=None, tr=[0.2, 0.2], t_actual_epoch=0.2,
    bit_epoch=-1, Elist=None, theta=None, loss=0, acc=0):  #这里的参数列表最好不要动（减少参数），因为fedavg也使用了这些参数，一旦改动那么fedavg也要改
        self.model = model
        self.id = id # integer
        self.group = group
        '''使用q-ffl请注释'''
        # self.train_data = {k: np.array(v) for k, v in train_data.items()}  #训练数据
        # self.eval_data = {k: np.array(v) for k, v in eval_data.items()}    #测试数据
        # self.num_samples = len(self.train_data['y'])
        # self.test_samples = len(self.eval_data['y'])

        self.ideal_epoch = ideal_epoch                                     #client能够完成的epoch数目
        self.actual_epoch = actual_epoch                                   #记录了上一轮服务器分配给client的epoch数目
        if (achieved == None):
            achieved = []
        self.achieved = achieved                                           #单个任务是否完成的01历史指示列表
        self.selected_times = selected_times                               #client被选中的次数
        self.r = r                                                         #单个任务的变化步长
        self.two_task = two_task                                           #任务区间（两个任务），task[0]为简单任务,task[1]为困难任务
        if(two_task_achieved == None):
            two_task_achieved = [[]for i in range (0, 2)]
        self.two_task_achieved = two_task_achieved                         #两个任务是否完成的01历史指示列表
        self.tr = tr                                                       #两个任务的变化步长,因为Easy和difficult的变化步长可能不一样，所以用一个列表来放两个步长
        self.t_actual_epoch = t_actual_epoch                               #记录区间任务中上一轮服务器分配给client的epoch数目
        
        self.bit_epoch = bit_epoch                                         #client每隔step个epoch就更新一次该位的参数，server与client联系时读该数作为任务区间的决策值之一

        if(Elist == None):                                                 #EMA的公式参数Vn=αVn-1 + (1-α)Ek,Elist存放历史的drop out的近似epoch
            Elist = []
        self.Elist = Elist

        if(theta == None):
            theta = []
        self.theta = theta                                                 #theta存放历史的阈值，阈值append在最后
        self.loss = loss                                                   #loss存放每个客户端的loss值
        self.acc = acc                                                     #acc存放每个客户端的精度
        # q-ffl:前80%作为训练集，中间10%作为验证集，最后10%作为测试集,不使用q-ffl的时候最后一行要注释掉，否则训练会出现bug
        data_x = train_data['x'] + eval_data['x']
        data_y = train_data['y'] + eval_data['y']
        self.train_data = {'x': data_x[:int(len(data_x)*0.8)], 
                            'y': data_y[:int(len(data_y)*0.8)]}

        self.val_data   = {'x': data_x[int(len(data_x)*0.8):int(len(data_x)*0.9)], 
                            'y': data_y[int(len(data_y)*0.8):int(len(data_y)*0.9)]}

        self.test_data  = {'x': data_x[int(len(data_x)*0.9):], 
                            'y': data_y[int(len(data_y)*0.9):]}
    
        self.train_samples = len(self.train_data['y'])
        self.val_samples = len(self.val_data['y'])

        #'''使用非q-ffl请注释'''
        self.eval_data = self.test_data
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.test_data['y'])              # 这个不注释直接运行FedAvg这种就会是tot_correct比test_sample还多
    
    def set_ideal_epoch(self, ideal_epoch):
        '''set epoch client can achieve'''
        self.ideal_epoch = ideal_epoch

    def get_ideal_epoch(self):
        '''get epoch client can achieve'''
        return self.ideal_epoch
    
    def set_actual_epoch(self, actual_epoch):
        '''set epoch client get actually'''
        self.actual_epoch = actual_epoch

    def get_actual_epoch(self):
        '''get epoch client get actually'''
        return self.actual_epoch
    
    def set_achieved(self, ac):
        self.achieved.append(ac)
    
    def get_achieved(self):
        return self.achieved

    def set_selected_times(self, selected_times):
        self.selected_times = selected_times
    
    def get_selected_times(self):
        return self.selected_times

    def add_Elist(self, E):    #向Elist中存放历史的drop out处的epoch
        self.Elist.append(E)
    
    def add_theta(self, theta):
        self.theta.append(theta)

    def set_r(self, r):  #r是调整的步长
        self.r = r

    def set_two_task(self, two_task):
        self.two_task = two_task
    
    def set_two_task_achieved(self, low_achieved, high_achieved):
        self.two_task_achieved[0].append(low_achieved)
        self.two_task_achieved[1].append(high_achieved)

    def set_tr(self, tr):
        '''
        tr is a list hasing two values in which tr[0] is the step of the easy one task and tr[1] is the step of the difficult one
        '''
        self.tr = tr
    
    def set_t_actual_epoch(self, t_actual_epoch):
        self.t_actual_epoch = t_actual_epoch

    def set_bit_epoch(self, bit_epoch):
        self.bit_epoch = bit_epoch

    def update_r(self):
        '''
        # #根据历史完成记录对r进行调整
        # 00: r -= 0.2, epoch = epoch/2
        # 01: r += 0, epoch = epoch
        # 10: r -= 0.1, epoch -= 0.1
        # 11: r += 0.1, epoch += 0.1 
        update_r()也可以通过调用update_r_helper()来完成
        '''
        
        v_r, r = self.update_r_helper(self.r, self.achieved)
        self.set_r(r)

        return v_r

    def get_r(self):
        return self.r

    def update_tr(self):    #更新区间任务的tr
        '''
        # #根据历史完成记录对r进行调整
        # 00: r -= 0.2, epoch = epoch/2
        # 01: r += 0.0, epoch = epoch
        # 10: r -= 0.1, epoch -= 0.1
        # 11: r += 0.1, epoch += r
        # 返回的v_r0和v_r1是epoch真正应该加的数字,r0和r1则是当前的r值
        '''
        r0, r1 = self.tr[0], self.tr[1]
        v_r0, r0 = self.update_r_helper(r0, self.two_task_achieved[0])   
        v_r1, r1 = self.update_r_helper(r1, self.two_task_achieved[1])

        self.set_tr([r0, r1])

        return v_r0, v_r1

    def update_r_helper(self, r, achieved):   #一个辅助函数帮助计算其中一个r
        '''
        r为传进来要改变的r的初始值，achieved则是改变的历史依据
        返回值为实际要加的值和r的实际值(例如r实际变成了0.1但是epoch要保持不变，这时返回值则为0，0.1)
        '''
        '''
        # #根据历史完成记录对r进行调整
        # 00: r -= 0.2, epoch = epoch/2
        # 01: r += 0, epoch = epoch
        # 10: r -= 0.1, epoch -= 0.1
        # 11: r += 0.1, epoch += r 
        '''
        if(len(achieved) < 2):       #首先判断client的选中次数是否不足2
            return r, r
        
        second_last = achieved[-2]   #取倒数第二次的完成情况
        last = achieved[-1]          #取最近一次的完成情况

        if(second_last == 0 and last == 0):
            r -= 0.2                 #允许r<0
            # if(self.r > 0.2): self.r -= 0.2
            # else:
            #     self.r = 0
            return 0, r
        elif(second_last == 0 and last == 1):
            r += 0
            return 0, r
        elif(second_last == 1 and last == 0):
            # self.r -= 0.1
            if(r > 0.1): r -= 0.1
            else:
                r = 0
            return -0.1, r
        elif(second_last == 1 and last == 1):
            r += 0.1
            return r, r

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, model_len)
    
    #q-ffl
    def get_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return loss


    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size                                       #计算接收到的模型字节数
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples                      #为什么要用flops*num_samples
        bytes_r = self.model.size                                       #计算传输的模型字节数，这两者应该是用于通信量的计算
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=1, batch_size=10):                 #计算num_epoch个epoch
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_inner(self.train_data, num_epochs, batch_size)    #获得局部最优解和计算量
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def solve_iters(self, num_iters=1, batch_size=10):                                  #计算几个iteration
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_iters(self.train_data, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    def solve_entire(self, num_epochs=1, num_iters=1, batch_size=10):
        bytes_w = self.model.size
        soln, comp = self.model.solve_entire(self.train_data, num_epochs, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def solve_entire_segment_upload(self, upload_crash, num_epochs=1, num_iters=1, batch_size=10, segment=0): 
        bytes_w = self.model.size
        solns, comp = self.model.solve_entire_segment_upload(self.train_data, upload_crash, num_epochs, num_iters, batch_size, segment)
        bytes_r = self.model.size
        if(segment > 0):
            samples_solns = [(self.num_samples, soln) for soln in solns]
            return samples_solns, (bytes_w, comp, bytes_r)
        else:
            return (self.num_samples, solns), (bytes_w, comp, bytes_r)

    def solve_entire_segment_upload_keeplast(self, upload_crash, num_epochs=1, num_iters=1, batch_size=10, segment=0): 
        bytes_w = self.model.size
        solns, comp = self.model.solve_entire_segment_upload_keeplast(self.train_data, upload_crash, num_epochs, num_iters, batch_size, segment)
        bytes_r = self.model.size
        if(segment > 0):
            samples_solns = [(self.num_samples, soln) for soln in solns]
            return samples_solns, (bytes_w, comp, bytes_r)
        else:
            return (self.num_samples, solns), (bytes_w, comp, bytes_r)

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)                          #mclr.py中的test函数计算分类正确的样本个数和交叉熵损失
        #self.loss = loss
        return tot_correct, loss, self.num_samples


    def test(self):                                                                   #client中定义的test()计算测试集分类正确的样本数和损失
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.eval_data)                           #通过调用mclr等模型中的test()开始训练
        # print("in client tot_correct: ", tot_correct, " test_samples: ", self.test_samples)
        return tot_correct, self.test_samples

    def get_num_classes(self):
        '''
        get the number of classes of this client
        '''
        num_classes = 0
        label = self.train_data['y']

        for y in label:

            num_classes += 1

        return num_classes
