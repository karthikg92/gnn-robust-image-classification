from utils import *
import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class DataProcessor:
	def __init__(self, nodes=25, k=8, batchsize=25, A_random=False):
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		self.N = nodes # Nodes in the graph
		self.k = k # k- nearest neighbor
		self.BatchSize = batchsize
		self.A_random = A_random
		self.index, self.A = image2graph_setup(nodes,k)
		self.DataPoints = self.mnist.train.num_examples

	def next_train(self):
		image, label = self.mnist.train.next_batch(self.BatchSize)
		f = compute_f( image.reshape([-1,28,28,1] ) , self.index)
		Adj_Batch = np.tile(self.A, (self.BatchSize,1,1)) # Replicating the same Adj matrix for all the input images
		return f, Adj_Batch, label


	def validation(self):
		image, label = self.mnist.test.images, self.mnist.test.labels
		f = compute_f( image.reshape([-1,28,28,1] ) , self.index)
		Adj = np.tile(self.A, (10000,1,1))
		return f, Adj, label