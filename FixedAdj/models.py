import tensorflow as tf
import layers as layers

def graph_cnn(f,A,Labels, learning_rate, training_flag, params):

	# Adding the graph layers
	for i in range(len(params.filters)):
		f = layers.GraphLayer(f, A, params.filters[i], 'GLayer'+str(i))
		if params.batch_norm:
			f = tf.layers.batch_normalization(f, axis=1 , training=training_flag)

	# Fully connected layer in the end
	f_flat = layers.flatten(f)
	#f_dropout=tf.nn.dropout(f_flat, 0.99) # Optional, not used in our experiments
	y = layers.fully_connected(f_flat, params.N_class)

	loss = layers.cross_entropy_loss(y,Labels)

	# Required to do this grouping again due to the additional variables introduced during BatchNorm
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	summary = tf.summary.merge_all()

	return train_step, accuracy, summary