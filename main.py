import numpy as np
import argparse
import importlib
import random
import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd+', 'fedprox_origin']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist', 
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is (F)EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ) # num_classes
}


#该函数用于fun_fedavg中的运行参数传参
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',             #名字或可选字符，可选参数由'-'标记，剩下的用作位置参数，大概是空格
                        help='name of optimizer;',
                        type=str,                  #type代表命令行参数应该被解析成什么类型，此处解析成字符
                        choices=OPTIMIZERS,        #choices代表可选的参数容器，optimizer的可选参数都放在了OPTIMIZER中
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)


    try: parsed = vars(parser.parse_args())              #将字符串参数转换成对象并将它们作为属性赋值给名称空间，返回的是一个名称空间，name=value这种形式
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])      #因为synthetic有iid、_0_0等多个种类，所有的合成数据集都是使用的flearn.models.synthetic.mclr
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model']) #其余数据集使用的模型可以直接根据flearn.models.数据集名字.模型名字索引获得模型的py文件

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')                        #获得mod的属性'Model'的值，learner代表模型，即mclr这种

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']  #FedAv、FedProx等优化器存放路径都为flearn.trainers.优化器名字
    mod = importlib.import_module(opt_path)                #加载优化器
    optimizer = getattr(mod, 'Server')                     #获得对应优化器的Server类，还可以使用getattr(mod, 'Server')()直接将这个类实例化

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]  #model_path按照.分割从下标2开始往后的所有即mnist.mclr这种

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);                            #获得最长参数名的长度，制表的时候要用到
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)           #按照键的字母顺序制表，但print('\t%11s : %s' % keyPair),这是什么意思

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data:                                                            #data文件夹下存放着各个数据集的训练集和测试集
    train_path = os.path.join('data', options['dataset'], 'data', 'train')  #nist的train数据的路径确实在data.nist.data.train下面
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    # call appropriate trainer
    t = optimizer(options, learner, dataset)            #实例化各优化器(如fedavg)中的Server类，参数是(params,learners, dataset)，options是接收到的输入参数
    t.train()                                           #调用Server中的train方法进行联邦学习客户端训练
    
if __name__ == '__main__':
    main()
