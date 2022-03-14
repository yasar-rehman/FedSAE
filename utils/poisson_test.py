#from matplotlib import lines
import numpy as np
import random
import os
import json
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm

# from flearn.models.client import Client
# from flearn.utils.model_utils import Metrics
# from flearn.utils.tf_utils import process_grad
from scipy.stats import beta
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from scipy.stats import poisson

# 第一个参数为mu
np.random.seed(0)
y = poisson.rvs(15, size = 1000)

# print(y)

x = np.arange(0, 1000)


# 折线图
# plt.plot(x, y)
# plt.savefig('./poisson.jpg')
# plt.show()
def draw_bar_poisson(y, path):
    # 条形图,x_pl代表数大小，y_pl统计该数频次
    # tolist将array转换成list
    y = y.tolist()
    x_pl = []
    y_pl = []
    for i in y:
        if(i not in x_pl):
            x_pl.append(i)
            y_pl.append(y.count(i))

    plt.bar(x_pl, y_pl)
    plt.savefig(path)
    plt.show()


#draw_bar_poisson(y, './bar_poisson.jpg')
#print("show!")