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
from sklearn import cluster

# 好像需要一个二维的array，可是epoch只是一维的
# 那我改成epoch和variance组成一个二维的
def hierarchical_lustering(vals):
    clst = cluster.AgglomerativeClustering(5)
    predicted_labels = clst.fit_predict(vals)
    print("labels_: ", clst.labels_)

    #计算每个类别分别有多少个样本
    dic = {}
    labels = clst.labels_.tolist()
    for i in labels:
        if(i not in dic):
            dic[i] = labels.count(i)
    # 好像层次聚类各个类别之间的样本数也并不平均
    print("statistical results of labels: ", dic)

vals = poisson.rvs(10, size=100)
variance = random.sample(range(0, 200), 100)
print(vals)

# 把两个一维数组组合成一个二维数组
out_list = [list(item) for item in zip(vals, variance)]
# print("after zip: ", out_list)

# hierarchical_lustering(out_list)
hierarchical_lustering(vals.reshape(-1, 1))
print("type of vals:", type(vals))
