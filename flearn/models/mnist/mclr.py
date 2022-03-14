import numpy as np
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''
    
    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()       #tf.train.Saver()在训练期间保存模型的checkpoint文件，以便于在训练中断时恢复模型继续训练，saver.save(sess,'/temp/model.ckpt')保存模型，saver.restore(sess,'')恢复模型
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)      #返回图的字节数
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())           #初始化全局变量
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()  #返回配置float操作的选项
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops  #返回浮点运算数，即计算花费
    
    def create_model(self, optimizer):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 784], name='features')   #占位符，feature，类型是tf.float32，二维张量None×784
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')             #正确的标签
        logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.001)) #全连接层，units输出的维度大小，改变input的最后一维，num_classes是传进来的参数，输出图片属于每个类别的概率
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),                             #axis=1表示按行，axis=0表示按列，按行返回输入的最大值的下标，代表的就是所属类别
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")           #返回向量每个类的概率
            }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) #计算交叉熵损失

        grads_and_vars = optimizer.compute_gradients(loss)                          #optimizer.compute_gradients()计算loss的梯度，返回值(梯度，变量)对即关于某变量的梯度
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())  #将compute_gradients()计算得到的梯度作为输入对权重进行更新
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))                  #计算分类正确的样本数目，如果预测值和labels相同则为非0，返回非0值的个数，但是怎么转换成准确率的
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:                                  #如果传进来的模型参数不为空则给模型参数重新赋值，如果传进来的是空呢？
            with self.graph.as_default():                             #返回一个上下文管理器，上下文管理器使用这个图作为默认的图
                all_vars = tf.trainable_variables()                   #获得一个可训练的变量列表
                for variable, value in zip(all_vars, model_params):   #将变量和变量的值打包在一起
                    variable.load(value, self.sess)                   #从图中给各变量赋值，不产生额外的操作

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())    #运行啥，获得可训练变量的参数值吗
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)   #初始化梯度全为0
        num_samples = len(data['y'])  #样本数目

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,                             #grads是create_model中计算出来的关于各个变量的梯度
                feed_dict={self.features: data['x'], self.labels: data['y']})   #这里的model_grads是什么，为什么要用sess.run()
            grads = process_grad(model_grads)                                   #返回一个展平的梯度

        return num_samples, grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):                  #solve_inner计算的是epoch，一个epoch有多个iteration
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):   #trange()相当于range()但是这里不知道在怎么用
            for X, y in batch_data(data, batch_size):                          #获得数据集的所有batch
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})          #可能是执行训练操作，feed的数据是features和labels，train_op是将计算的梯度更新到变量
        soln = self.get_params()                                               #soln是计算得到的solution？
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops  #comp统计有多少计算量，epoch数×所有batch的个数×每个batch样本数×每个样本的浮点运算数
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):                   #solve_iters计算的是迭代次数
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):    #获得num_iters个batch
            with self.graph.as_default():                                      #执行训练
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = 0                                                               #为什么这里comp直接是0，也可以计算 num_iters * batch_size * self.flops
        return soln, comp
    
    def solve_entire(self, data, num_epochs=1, num_iters=1, batch_size=32):    #训练精确到小数的epoch
        if(num_iters > 0):
            for X, y in batch_data_multiple_iters(data, batch_size, num_iters):    #获得num_iters个batch
                with self.graph.as_default():                                      #执行训练
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):   #trange()相当于range()但是这里不知道在怎么用
            for X, y in batch_data(data, batch_size):                          #获得数据集的所有batch
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})          #可能是执行训练操作，feed的数据是features和labels，train_op是将计算的梯度更新到变量
        soln = self.get_params()                                               #soln是计算得到的solution？
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops  #comp统计有多少计算量，epoch数×所有batch的个数×每个batch样本数×每个样本的浮点运算数
        return soln, comp

    def solve_entire_segment_upload(self, data, upload_crash, num_epochs=1, num_iters=1, batch_size=32, segment=0):           #运行精确到小数的epoch并分段上传
        solns = []
        if(num_iters > 0):
            for X, y in batch_data_multiple_iters(data, batch_size, num_iters):    #获得num_iters个batch
                with self.graph.as_default():                                      #执行训练
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        
                                               #计算隔几个epoch上传
        for i in trange(1, num_epochs+1, desc='Epoch: ', leave=False, ncols=120):   #trange()相当于range()但是这里不知道在怎么用
            for X, y in batch_data(data, batch_size):                          #获得数据集的所有batch
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})          #可能是执行训练操作，feed的数据是features和labels，train_op是将计算的梯度更新到变量
            if(segment>0 and (i in upload_crash or i%math.ceil(segment) == 0)):
                solns.append(self.get_params())
        soln = self.get_params()                                               #运行完了获得soln的参数,仅仅是单个soln而不是分段的结果
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops  #comp统计有多少计算量，epoch数×所有batch的个数×每个batch样本数×每个样本的浮点运算数
        if(segment > 0):      #segment>0说明需要进行分段上传，返回的是分段值
            return solns, comp
        return soln, comp


    #分段上传并保留最后一次上传的段
    def solve_entire_segment_upload_keeplast(self, data, upload_crash, num_epochs=1, num_iters=1, batch_size=32, segment=0):           #运行精确到小数的epoch并分段上传
        solns = []
        if(num_iters > 0):
            for X, y in batch_data_multiple_iters(data, batch_size, num_iters):    #获得num_iters个batch
                with self.graph.as_default():                                      #执行训练
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        
        #计算隔几个epoch上传
        for i in trange(1, num_epochs+1, desc='Epoch: ', leave=False, ncols=120):   #trange()相当于range()但是这里不知道在怎么用
            for X, y in batch_data(data, batch_size):                          #获得数据集的所有batch
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})          #可能是执行训练操作，feed的数据是features和labels，train_op是将计算的梯度更新到变量
            if(segment>0 and (i in upload_crash or i%math.ceil(segment) == 0)):
                if(len(solns) == 0):                                           #如果solns为空那么就往里面append
                    solns.append(self.get_params())
                else:                                                          #solns不为空那么就保留最后一个元素并append当前的值
                    solns = [ solns[-1] ]
                    solns.append(self.get_params())
                
        soln = self.get_params()                                               #运行完了获得soln的参数,仅仅是单个soln而不是分段的结果
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops  #comp统计有多少计算量，epoch数×所有batch的个数×每个batch样本数×每个样本的浮点运算数
        if(segment > 0):      #segment>0说明需要进行分段上传，返回的是分段值
            return solns, comp
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss], #eval_metric_ops为计算分类正确的图片个数，loss为计算交叉熵损失
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()
