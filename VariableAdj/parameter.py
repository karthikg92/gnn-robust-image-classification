import numpy as np

class Parameters():
    def __init__(self):
        self.N = 50
        self.N_class = None # Will be set later by the data processor
        self.k = 8
        self.LearningRate = self._constant_paramenter
        
        self.Epoch = 1000
        self.BatchSize = 256
        self.te_batch_size = 32
        self.filters = [5] 
        self.AdditionalFC = None #None, if default, else supply a list of natural numbers (# of neurons per FC layer)
        self.batch_norm = True # True or false

        # None or a value
        self.np_rand_seed = None
        self.tf_rand_seed = None

        
        self.gpu_fraction = None #None or a value
        self.gpu_growth = None #None or True

        self.summary_folder = 'summary_1'
        self.checkpoint_folder = 'checkpoint_1'
        self.summary_freq = 50
        self.print_freq = 50
        self.checkpoint_freq = 2500
        self.validation_freq = 500

    # Called after reading the data (since we may only use certain digits)
    def update_N_class(self, n):
        self.N_class = n
        return None

    # Schedule for learning rate
    def _ramp_generator(self, i,N_iter):
        l_lim = 5e-4
        h_lim = 5e-3
        if i<N_iter/2:
            return l_lim + (h_lim-l_lim)/N_iter*2*i
        else:
            return h_lim + (1-2*i/N_iter)*(h_lim-l_lim)

    def _constant_paramenter(self, i, N_iter):
        return 5e-4
