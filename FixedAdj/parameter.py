import numpy as np

class Parameters():
    def __init__(self):
        self.N= 40
        self.N_class= 10
        self.k=8
        self.LearningRate=self._constant_paramenter
        
        self.Epoch=100
        self.BatchSize=32
        self.te_batch_size=32
        self.filters = [4, 4]
        self.batch_norm = True

        # None or a value
        self.np_rand_seed = None
        self.tf_rand_seed = None

        #None or a value
        self.gpu_fraction = None
        self.gpu_growth = None

        # Checkpoint will store all variables to recover models
        # Summary will output state of training to tensorboard
        # Validation will write output on screen
        # Print will write training batch accuracy
        # All freqiencies are in terms of number of iterations
        self.summary_folder='summary_1'
        self.checkpoint_folder='checkpoint_1'
        self.summary_freq=2500
        self.print_freq=500
        self.checkpoint_freq=2500
        self.validation_freq=500

    
    # Functions that can vary the learning rate based on iteration
    
    def _ramp_generator(self, i,N_iter):
        l_lim=5e-4
        h_lim=5e-3
        if i<N_iter/2:
            return l_lim + (h_lim-l_lim)/N_iter*2*i
        else:
            return h_lim + (1-2*i/N_iter)*(h_lim-l_lim)

    def _constant_paramenter(self, i, N_iter):
        return 5e-4
