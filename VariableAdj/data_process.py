import numpy as np 
import tensorflow as tf
import pickle

class DataProcessor:
	def __init__(self, params):
		# function on nodes : f
		# adjacency matrix : A
		# label : l
		self.tr_f, self.tr_A, self.tr_l, self.te_f, self.te_A, self.te_l = self._load_dataset(params.N)		
		self.N = params.N
		self.BatchSize = params.BatchSize
		params.update_N_class( self.tr_l.shape[1] )
		self.DataPoints = len(self.tr_f)
        
	def _load_dataset(self, nodes):
		filehandler = open('MNIST_' + str(nodes) + '.pkl', 'rb')
		tr, te = pickle.load(filehandler)
		return np.expand_dims(tr[0], axis = 2 ), tr[1], tr[2], np.expand_dims(te[0] , axis = 2), te[1], te[2]

	def _get_index(self, k, N):
		a = [i for i in range(N)]
		a = np.random.permutation(a)
		return a[:k]
    
	def next_train(self):
		index = self._get_index( self.BatchSize, len(self.tr_f) )
		f = self.tr_f[index, :, :]
		A = self.tr_A[index, :, :]
		l = self.tr_l[index, :]
		return f, A, l


	def validation(self):
		return self.te_f, self.te_A, self.te_l
