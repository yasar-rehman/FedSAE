import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    def __init__(self, num_classes, optimizer, seed=1):
        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        """Model function for CNN."""
        features = tf.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        output2 = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='output2')
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))


        kl_loss = tf.keras.losses.KLD(predictions['probabilities'], output2) + tf.keras.losses.KLD(output2, predictions['probabilities'])
        kl_grads_and_vars = optimizer.compute_gradients(kl_loss)
        kl_grads, _ = zip(*kl_grads_and_vars)

        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
            # Len of model_grads (tuple) = 784+10
            grads = process_grad(model_grads)
        return num_samples, grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = 0
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
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()

    def reinitialize_params(self, seed):
        tf.set_random_seed(seed)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            model_params = self.get_params()
        return model_params