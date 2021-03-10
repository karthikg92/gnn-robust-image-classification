import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as sk

from parameter import *
from utils import *
from data_process import *
from models import graph_cnn
from trainer import Trainer

# Loading all parameters
params = Parameters()

if params.np_rand_seed is not None:
	np.random.seed(params.np_rand_seed)

if params.tf_rand_seed is not None:
	tf.set_random_seed(params.tf_rand_seed)

# Will supply training and testing data
mnist = DataProcessor(nodes=params.N, k=params.k, batchsize=params.BatchSize)

trainer = Trainer(mnist, params)
trainer.train()
