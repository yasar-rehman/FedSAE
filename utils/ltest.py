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

def test_np_choice():
    all_samples = [1, 3, 7, 9 ,5, 4.4, 8, 11, 23 , 21, 90, 34, 78, 98, 101]
    indices = [0, 1, 4, 5, 9, 7, 8, 11]
    prob = [0.01, 0.02, 0.05, 0.02, 0.01, 0.03, 0.07, 0.09]
    print("initial: ", np.array(prob)/np.float(sum(prob)), "sum: ", sum(np.array(prob)/np.float(sum(prob))))

    # p的总和要等于1
    value = np.random.choice(indices, 4, p=np.array(prob)/np.float(sum(prob)), replace=False)
    print(value)

 
test_np_choice()   