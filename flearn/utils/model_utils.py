import json
import numpy as np
import os

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()  #np.random.get_state()设定状态，记录下数组被打乱的操作
    np.random.shuffle(data_x)          #打乱数组data_x，即图片
    np.random.set_state(rng_state)     #接收get_state()的返回值，进行同样的打乱操作
    np.random.shuffle(data_y)          #data_y标签和图片进行同样的shuffle操作，之后仍然是一一对应

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):  #获得所有批数据
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)             #return是函数结束，yield是暂停，下一次运行的时候又从yield的下一步开始，即下一次又接着生成一个batch

def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    idx = 0                                #idx代表batch的起始下标

    for i in range(num_iters):
        if idx+batch_size >= len(data_x):  #如果该批次数据的样本数量大于了总的数据量，那么剩下的所有数据自成一个batch
            idx = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
        batched_x = data_x[idx: idx+batch_size]
        batched_y = data_y[idx: idx+batch_size]
        idx += batch_size
        yield (batched_x, batched_y)
#q-ffl
def gen_batch(data, batch_size, num_iter):
    data_x = data['x']
    data_y = data['y']
    index = len(data_y) 

    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data_y)):
            index = 0
            np.random.seed(i+1)
            # randomly shuffle the data after one pass of the entire training set         
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        batched_x = data_x[index: index + batch_size]
        batched_y = data_y[index: index + batch_size]
        
        yield (batched_x, batched_y)
#q-ffl
def gen_epoch(data, num_iter):
    '''
    input: the training data and number of iterations
    return: the E epoches of data to run gradient descent
    '''
    data_x = data['x']
    data_y = data['y']
    for i in range(num_iter):
        # randomly shuffle the data after each epoch
        np.random.seed(i+1)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

        batched_x = data_x
        batched_y = data_y

        yield (batched_x, batched_y)


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []                                                    #最终存放类型为字符串"f_00000", "f_00001"
    groups = []                                                     #组号
    train_data = {}                                                 #存放类型为键值对 {"f_00001":{"y":[0,1,3,...],"x":[[],[],[],...]}}
    test_data = {}

    train_files = os.listdir(train_data_dir)                        #os.listdir()返回指定路径下的文件或者文件夹列表
    train_files = [f for f in train_files if f.endswith('.json')]   #以.json结尾的则为正确的训练文件，因为train下面还有以.ipynb_checkpoints结尾的文件
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)                  #组合出正确的训练集文件的路径，文件夹路径加上训练文件全称
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)                                  #加载json文件  "users":["f_00000", "f_00001",...]
        clients.extend(cdata['users'])                              #cdata['users']将会被添加到列表clients的末尾，则clients[]存放客户端的id
        if 'hierarchies' in cdata:                                  #如果客户端数据集cdata中有关于等级的信息则分组
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])                       #user_data中存放的是键值对"user_data":{"f_00101":{"y":[], "x":[[],[]]}}

    test_files = os.listdir(test_data_dir)                          #加载测试集数据
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))                       #clients取训练数据中的键并有序排列，前面已经使用了cdata['users']填充，为什么后面又要重新从训练数据中的键来取

    return clients, groups, train_data, test_data                   #clients到底是什么类型的，为什么使用的时候都是当做的integer类型在使用


class Metrics(object):                                                          #评估类
    def __init__(self, clients, params):                                        #析构函数，初始化各参数
        self.params = params
        num_rounds = params['num_rounds']                                       #num_rounds为训练的轮数，传参的时候在.sh文件中传进来
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}          #一个字典，键为c.id 值为[0,0,0,...]里面有num_rounds个0，存放每轮写的字节数
        self.client_computations = {c.id: [0] * num_rounds for c in clients}    #c.id是个什么东西，遍历的c作为clients的元素不是 "f_00001"这种吗，加个id是什么
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies = []                                                    #accuracies存放测试集精度
        self.train_accuracies = []                                              #train_accuracies存放训练集精度

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats                                          #stats是一个tuple，所以可能是(a,b,c)这样的，然后赋值给bytes_w，comp，bytes_r
        self.bytes_written[cid][rnd] += bytes_w                                 #bytes_written[用户id][轮数?] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']                       #前面这些参数都是通过传进来的params得到
        metrics['accuracies'] = self.accuracies                                 #accuracies等参数通过每轮的计算得到
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        #这一步是在拼接地址 "out.mnist.metrics.seed..."但是我没有找到这样的系统目录
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
	    #os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists(os.path.join('out', self.params['dataset'])):     #如果输出文件路径out.mnist不存在则创建
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)                                             #序列化metric作为json形式的流到ouf中，即把metric写到某文件中
