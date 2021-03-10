import tensorflow as tf

# All variables created with Xavier Initialization
xav_initializer = tf.contrib.layers.xavier_initializer()

def Identity(n):
    with tf.name_scope("Identity"):
        I=tf.eye(n)
        return I

def bias_var(shape, name='bias'):
    with tf.name_scope(name):
        #b = tf.Variable(tf.truncated_normal(shape, stddev=0.2))
        b = tf.Variable(xav_initializer(shape))
        with tf.name_scope('summaries'):
            tf.summary.histogram('BiasVar', b)
        return b   

def weight_var(shape, name='Weight'):
    with tf.name_scope(name):
        #w=tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        w = tf.Variable(xav_initializer(shape))
        return w

def initH(A, name='H'):
    with tf.name_scope(name):
        b1 = bias_var([1], name='a1')
        b2 = bias_var([1], name='a2')
        N = int(A.get_shape()[2])
        x = tf.add( tf.multiply( b1 , Identity(N) ) , b2*A )
        return x

def GraphLayer(f_in, A, F, name='GraphLayer'):
    with tf.name_scope(name):
        # The number of input features is C and output features is F
        C = int(f_in.get_shape()[2])
        output_layer=[]
        for i in range(F):
            V_f = tf.add_n([tf.matmul(initH(A), tf.expand_dims(f_in[:,:,c], axis=2)) for c in range(C)]) + bias_var([1])
            output_layer.append(V_f)    
        f_out = tf.squeeze(tf.stack(output_layer, axis=2), [3])
        f_relu = tf.nn.relu(f_out)
        print(name + " added")
        return f_relu
    

def flatten(x):
    with tf.name_scope('flatten'):
        dim_flat = int(x.get_shape()[1]) * int(x.get_shape()[2])
        y = tf.reshape(x, shape=[-1,dim_flat])
        return y

def fully_connected(x_in, M, name='FullyConnected'):
    with tf.name_scope(name):
        N = int(x_in.get_shape()[1])
        W_fc = weight_var([N, M])
        b_fc = bias_var([M])
        x_out = tf.matmul(x_in, W_fc) + b_fc
        x_relu = tf.nn.relu(x_out)
        return x_relu

def cross_entropy_loss(x, labels):
    cross_entropy= tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=x, name='CrossEnt'))
    with tf.name_scope('summaries'):
        tf.summary.scalar('CE_Loss', cross_entropy)
    return cross_entropy