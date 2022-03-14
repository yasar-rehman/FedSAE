import numpy as np
import random
import os
import json
import math
#from utils.poisson_test import draw_bar_poisson
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
from scipy.stats import beta
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn import cluster

mu = np.random.uniform(-1, 60)
num_ran = [np.random.uniform(-1, 60) for i in range]
print(num_ran)