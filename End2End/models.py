import tensorflow as tf
import layers as layers

def graph_cnn(f,A,Labels, learning_rate, training_flag, params):

	for i in range(len(params.filters)):
		f=layers.GraphLayer(f, A, params.filters[i], 'GLayer'+str(i))
		if params.batch_norm:
			f = tf.layers.batch_normalization(f, axis=1 , training=training_flag)

	f_flat=layers.flatten(f)
	#f_dropout=tf.nn.dropout(f_flat, 0.99)

	if params.AdditionalFC is None:
		y = layers.fully_connected(f_flat, params.N_class)
	else:

		y = layers.fully_connected(f_flat, params.AdditionalFC[0])
		if len(params.AdditionalFC)>1:
			for i in range(1, len(params.AdditionalFC)):
				y = layers.fully_connected(y, params.AdditionalFC[i])
		y = layers.fully_connected(y , params.N_class)



	loss=layers.cross_entropy_loss(y,Labels)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	summary=tf.summary.merge_all()

	return train_step, accuracy, summary


def e2e_v1(Images, Labels, learning_rate, training_flag, params):

	# Go from an image to an A and f
	for i in range(len(params.Cfilters)):
		Images = conv(Images, params.Cfilters[i], name = 'ConvLayer' + str(i))
		Images = max_pool(Images, name = "MaxPool" + str(i))

	f = Image2f(Images)
	A = Image2A(Images)

	# Need an f and A
	for i in range(len(params.Gfilters)):
		f = layers.GraphLayer(f, A, params.Gfilters[i], 'GLayer'+str(i))
		if params.batch_norm:
			f = tf.layers.batch_normalization(f, axis=1 , training=training_flag)

	f_flat=layers.flatten(f)
	#f_dropout=tf.nn.dropout(f_flat, 0.99)

	if params.AdditionalFC is None:
		y = layers.fully_connected(f_flat, params.N_class)
	else:

		y = layers.fully_connected(f_flat, params.AdditionalFC[0])
		if len(params.AdditionalFC)>1:
			for i in range(1, len(params.AdditionalFC)):
				y = layers.fully_connected(y, params.AdditionalFC[i])
		y = layers.fully_connected(y , params.N_class)

	loss = layers.cross_entropy_loss(y,Labels)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	summary = tf.summary.merge_all()

	return train_step, accuracy, summary
