from utils import *
import numpy as np 
import tensorflow as tf
import pickle

class DataProcessor:
	def __init__(self, params):
		self.tr_img, self.tr_lab, self.te_img, self.te_lab = self._load_dataset(params.fname)		
		self.BatchSize = params.BatchSize
		params.update_N_class( self.tr_lab.shape[1] )
		self.DataPoints = self.tr_img.shape[0]

	def _load_dataset(self, fname):
		filehandler = open( fname, 'rb')
		tr_img, tr_lab, te_img, te_lab = pickle.load(filehandler)
		tr_img = np.reshape( tr_img, [-1, 28, 28, 1])
		te_img = np.reshape( te_img, [-1, 28, 28, 1])
		return tr_img, tr_lab, te_img, te_lab

	def _get_index(self, k, N):
		a = [i for i in range(N)]
		a = np.random.permutation(a)
		return a[:k]
    
	def next_train(self):
		index = self._get_index( self.BatchSize, self.te_img.shape[0] ) 
		img = self.tr_img[index, :, :]
		lab = self.tr_lab[index, :]
		return img, lab

	def validation(self):
		return self.te_img, self.te_lab