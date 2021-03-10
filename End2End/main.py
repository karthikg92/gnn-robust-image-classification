import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as sk

from parameter import *
from data_process import *
from models import graph_cnn
from trainer import Trainer

params=Parameters()

if params.np_rand_seed is not None:
	np.random.seed(params.np_rand_seed)

if params.tf_rand_seed is not None:
	tf.set_random_seed(params.tf_rand_seed)

print('Reading data...')
mnist = DataProcessor( params )
print('Data processing complete...')

trainer=Trainer(mnist, params)
print('Trainer set up complete...')
trainer.train()
