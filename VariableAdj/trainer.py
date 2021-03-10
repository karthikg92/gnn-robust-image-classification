import tensorflow as tf
from models import graph_cnn
import os as os
import numpy as np

class Trainer():

    def __init__(self, data, params):
        self.f = tf.placeholder(tf.float32, [None, params.N,1])
        self.A = tf.placeholder(tf.float32, [None, params.N, params.N])
        self.Labels = tf.placeholder(tf.float32, [None, params.N_class])
        self.training_flag = tf.placeholder(tf.bool)
        self.learning_rate = tf.placeholder(tf.float32)
        self.data = data
        self.params = params
        self.lr_function= params.LearningRate

        self.train_step, self.accuracy, self.summary= graph_cnn(self.f , self.A , self.Labels, self.learning_rate , \
         self.training_flag , params)

        self.summary_writer = tf.summary.FileWriter(params.summary_folder)
        self.checkpoint = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

        # Validation
        self.f_val, self.A_val, self.label_val = self.data.validation()
        self.best_val=0

    def _initialize_session(self):
        if self.params.gpu_fraction is not None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.params.gpu_fraction
            self.sess=tf.Session(config=config)
        elif self.params.gpu_growth is not None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess=tf.Session(config=config)
        else:
            self.sess=tf.Session()
        return None


    def _initialize_train(self):
        self._initialize_session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer.add_graph(self.sess.graph)
        self.checkpoint_path=os.getcwd()+ '/'+ self.params.checkpoint_folder+ '/tf_checkpoint'
        os.makedirs( self.checkpoint_path, exist_ok=True)
        self.checkpoint.save(self.sess, self.checkpoint_path)
        return None

    def train(self):
        self._initialize_train()
       
        N_iter= self.params.Epoch* self.data.DataPoints//self.params.BatchSize

        for i in range(N_iter):
            f_batch ,A_batch , label_batch= self.data.next_train()
            current_learn_rate=self.lr_function(i, N_iter)
            input_dict = {self.f: f_batch, self.A: A_batch, self.Labels: label_batch, self.training_flag: True, self.learning_rate: current_learn_rate}
            self.sess.run(self.train_step, feed_dict=input_dict)
            
            if i% self.params.print_freq==0:
                acc_batch= self.sess.run(self.accuracy, feed_dict= input_dict)
                print("Epoch : {0}, Iteration: {1} , Tr batch acc = {2:.2f}".format(i//(self.data.DataPoints //self.params.BatchSize) +1, i, acc_batch) )

            if i% self.params.summary_freq==0:
                merged_s=  self.sess.run(self.summary, feed_dict= input_dict )
                self.summary_writer.add_summary(merged_s,i)

            if i%self.params.checkpoint_freq==0:
                self.checkpoint.save(self.sess, self.checkpoint_path)
                
            if i%self.params.validation_freq==0:
                acc=self._test()
                print("\n------------------Validation Data ----------------------")
                print("Epoch : {0}, Iteration: {1} , Val acc = {2:.2f}".format(i//(self.data.DataPoints //self.params.BatchSize) +1, i, acc) )
                print("--------------------------------------------------------\n")

        print("\n-----------------------")
        print("-----------------------")
        print("Best validation accuracy = {0:.2f}".format(self.best_val))
        print("-----------------------")
        print("-----------------------")

        return None


    def _test(self):
        te_batch_size=self.params.te_batch_size
        N_batch=self.f_val.shape[0]//te_batch_size
        acc_list=[]
        for i in range(N_batch):
            start=i*te_batch_size
            end=start+te_batch_size
            val_input_dict={self.f: self.f_val[start:end, :, :], self.A: self.A_val[start:end, :,:], self.Labels: self.label_val[start:end,:], self.training_flag: False, self.learning_rate: 1e-3}
            
            acc_list.append(self.sess.run(self.accuracy, feed_dict= val_input_dict))

        mean_accuracy=np.mean(acc_list)

        if mean_accuracy>self.best_val:
            self.best_val=mean_accuracy
        return mean_accuracy