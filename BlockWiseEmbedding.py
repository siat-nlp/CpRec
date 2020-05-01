import tensorflow as tf
import numpy as np


class BlockWiseEmbeddingForInput(object):
	"""
	# input shape
		2D tensor with shape：[batch_size, seq_len]
	# return
		3D tensor with shape：[batch_size, seq_len, output_dim]
	"""
	def __init__(self, vocab_size, embed_dim, block=None, block_factor = 4, blocks_dims=None, name=None):
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.block = block
		self.block_num = len(block)-1
		self.block_factor = block_factor

	def build(self):
		if self.block_factor == 1:
			stdv = np.sqrt(1. / self.vocab_size)
			self.embedding = tf.get_variable("word_embedding_", [self.vocab_size, self.embed_dim],
			                                 initializer=tf.random_uniform_initializer(-stdv, stdv))
		else:
			otherblock_dims = []
			block_factor = self.block_factor
			for i in range(self.block_num):
				dim = max(1, self.embed_dim / block_factor)
				otherblock_dims.append(dim)
				block_factor *= self.block_factor

			firstblock_K = self.block[0]
			self.firstblock_w = tf.get_variable("blockwiseembedding_block1_w", [firstblock_K, self.embed_dim])
			self.otherblock_w = []
			for i in range(self.block_num):
				block_i_dim = otherblock_dims[i]
				block_i_K = self.block[i + 1] - self.block[i]
				self.otherblock_w.append([
					tf.get_variable("blockwiseembedding_block{}_proj_w".format(i + 2), [block_i_dim, self.embed_dim]),
					tf.get_variable("blockwiseembedding_block{}_w".format(i + 2), [block_i_K, block_i_dim])
				])

	def get_input(self, inputs):
		# print("shape: ", inputs.shape)
		# inputs: [batch_size, seq_len]
		if self.block_factor == 1:
			outputs = tf.nn.embedding_lookup(self.embedding, inputs)
			print('using embeddding')
		else:
			input_size = list(inputs.shape)
			print("input_size: ", input_size)
			outputs = tf.zeros(input_size+[self.embed_dim], dtype=tf.float32)

			block_value = [0] + self.block
			for i in range(len(block_value) - 1):
				low_idx = block_value[i]
				high_idx = block_value[i+1]
				mask = tf.logical_and(tf.greater_equal(inputs, low_idx), tf.less(inputs, high_idx))

				# row_indices = tf.squeeze(tf.where(mask))
				mask = tf.cast(mask, dtype=float)

				# if row_indices.size() ==0:
				# 	continue
				if i == 0:
					firstblock_inputs = (inputs-low_idx)*tf.cast(mask, dtype=tf.int32)
					# firstblock_inputs = tf.boolean_mask(inputs-low_idx, mask)
					firstblock_embed = tf.nn.embedding_lookup(self.firstblock_w, firstblock_inputs)
					projected = firstblock_embed # [batch_size, seq_len, output_dim]
				else:
					# block_i_inputs = tf.boolean_mask(inputs-low_idx, mask)
					block_i_inputs = (inputs-low_idx)*tf.cast(mask, dtype=tf.int32)
					block_i_embed = tf.tensordot(tf.nn.embedding_lookup(self.otherblock_w[i-1][1], block_i_inputs), self.otherblock_w[i-1][0], axes=1)
					# tf.where(mask, block_i_embed, outputs)
					projected = block_i_embed
				outputs += projected*tf.expand_dims(mask, axis=-1)

		return outputs


class BlockWiseEmbeddingForSoftmax(object):
	"""
		make sure input_dim == embed_dim
	"""

	def __init__(self, input_dim, block, block_factor=4, otherblock_dims=None, dropout=0, adaptive_inputs=None,
	             initializer=None, name=None, tied_pro=True):
		self.block_num = len(block) - 1
		self.dropout = dropout
		self.block_factor = block_factor

		if otherblock_dims:
			assert (len(otherblock_dims) == self.block_num)
		else:
			otherblock_dims = []
			block_factor = self.block_factor
			for i in range(self.block_num):
				dim = max(1, input_dim / block_factor)
				otherblock_dims.append(dim)
				block_factor *= block_factor

		self.block = block
		firstblock_K = block[0] + self.block_num
		self.otherblock_w = []
		with tf.variable_scope(name or type(self).__name__, initializer=initializer):
			self.firstblock_w = tf.get_variable("blockwiseembedding_softmax_block1_w", [input_dim, firstblock_K])
			for i in range(self.block_num):
				block_i_dim = otherblock_dims[i]
				block_i_K = block[i + 1] - block[i]
				self.otherblock_w.append([
					tf.get_variable("blockwiseembedding_softmax_block{}_proj_w".format(i + 2), [input_dim, block_i_dim]),
					# dropout
					tf.get_variable("blockwiseembedding_softmax_block{}_w".format(i + 2), [block_i_dim, block_i_K])
				])
		print("dropout: ", dropout)

	def loss(self, inputs, labels, train=True, name='loss'):
		# Get block_i masks and update firstblock labels
		# start_time = time.time()

		training_losses = []
		if train:
			inputs = tf.nn.dropout(inputs, keep_prob=1 - self.dropout)

		firstblock_labels = labels
		ones = tf.ones([tf.size(labels)], dtype=tf.int32)
		for i in range(self.block_num):
			mask = tf.logical_and(tf.greater_equal(labels, self.block[i]), tf.less(labels, self.block[i + 1]))

			# Update firstblock labels
			firstblock_labels = tf.where(mask, ones * (self.block[0] + i), firstblock_labels)

			# Compute block_i loss
			block_i_inputs = tf.boolean_mask(inputs, mask)  # [block_i_num, channel]
			if train:
				block_i_logits = tf.matmul(
					tf.nn.dropout(tf.matmul(block_i_inputs, self.otherblock_w[i][0]), keep_prob=1 - self.dropout),
					self.otherblock_w[i][1])  # [block_i_num, block_i_dim]
			else:
				block_i_logits = tf.matmul(tf.matmul(block_i_inputs, self.otherblock_w[i][0]),
				                        self.otherblock_w[i][1])  # [block_i_num, block_i_dim]
			block_i_labels = tf.boolean_mask(labels - self.block[i], mask)  # [block_i_num]
			block_i_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=block_i_logits, labels=block_i_labels)
			training_losses.append(block_i_loss)
			aligned_block_i_loss = tf.SparseTensor(tf.squeeze(tf.where(mask)), block_i_loss,
			                                    [tf.size(labels, out_type=tf.int64)])
			# print("##################", tf.squeeze(tf.where(mask)).shape, block_i_loss.shape, [tf.size(labels, out_type=tf.int64).shape])
			loss = tf.sparse_tensor_to_dense(aligned_block_i_loss) if i == 0 else loss + tf.sparse_tensor_to_dense(
				aligned_block_i_loss)

		# Compute firstblock loss
		firstblock_logits = tf.matmul(inputs, self.firstblock_w)  # (sample_num, firstblock_size)
		firstblock_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=firstblock_logits,
		                                                           labels=firstblock_labels)  # (sample_num)
		training_losses.append(firstblock_loss)
		loss = tf.add(loss, firstblock_loss, name=name)

		# print("Adaptive_Softmax time: ", time.time() - start_time)
		return loss, training_losses

	def softmax(self, inputs, name='softmax'):
		firstblock_logits = tf.matmul(inputs, self.firstblock_w)
		firstblock_softmax = tf.nn.softmax(firstblock_logits)
		softmax_list = [firstblock_softmax[:, :self.block[0]]]
		for i in range(self.block_num):
			block_i_logits = tf.matmul(tf.matmul(inputs, self.otherblock_w[i][0]), self.otherblock_w[i][1])
			block_i_softmax = tf.nn.softmax(block_i_logits)
			index = self.block[0] + i
			softmax_list.append(block_i_softmax * firstblock_softmax[:, index:index + 1])
		return tf.concat(softmax_list, axis=1, name=name)

	def block_i_top_v(self, inputs, firstblock_softmax, i, top_v=5):
		block_i_logits = tf.matmul(tf.matmul(inputs, self.otherblock_w[i][0]), self.otherblock_w[i][1])
		block_i_softmax = tf.nn.softmax(block_i_logits)
		index = self.block[0] + i
		block_i_pro = block_i_softmax * firstblock_softmax[:, index:index + 1]
		block_i_top_value, block_i_top_indices = tf.nn.top_k(block_i_pro, k=top_v)
		# block_i_top = list((block_i_top_value[0][i], block_i_top_indices[0][i]) for i in range(top_v))
		block_i_top = list((block_i_top_value[0][j], block_i_top_indices[0][j] + self.block[i]) for j in range(top_v))
		return block_i_top

	def softmax_inference_top(self, inputs, name='softmax', top_v=5):
		firstblock_logits = tf.matmul(inputs, self.firstblock_w)  # [batch_size==1, firstblock_item]
		firstblock_softmax = tf.nn.softmax(firstblock_logits)
		temp_top_value, temp_top_indices = tf.nn.top_k(firstblock_softmax[:, :self.block[0]], k=top_v)
		temp_top = list((temp_top_value[0][i], temp_top_indices[0][i]) for i in range(top_v))

		for i in range(self.block_num):
			top_other = tf.nn.in_top_k(firstblock_softmax, [self.block[0] + i], k=top_v)
			block_i_top = tf.cond(tf.equal(top_other[0], tf.constant(True)),
			                   true_fn=lambda: self.block_i_top_v(inputs, firstblock_softmax, i, top_v), false_fn=lambda: list(
					(tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.int32)) for i in range(top_v)))
			temp_top.extend(block_i_top)

		return temp_top

	def log_softmax(self, inputs, name='log_softmax'):
		firstblock_logits = tf.matmul(inputs, self.firstblock_w)
		firstblock_logsoftmax = tf.nn.log_softmax(firstblock_logits)
		logsoftmax_list = [firstblock_logsoftmax[:, :self.block[0]]]
		for i in range(self.block_num):
			block_i_logits = tf.matmul(tf.matmul(inputs, self.otherblock_w[i][0]), self.otherblock_w[i][1])  # drouput
			block_i_logsoftmax = tf.nn.log_softmax(block_i_logits)
			index = self.block[0] + i
			logsoftmax_list.append(block_i_logsoftmax + firstblock_logsoftmax[:, index:index + 1])
		return tf.concat(logsoftmax_list, axis=1, name=name)