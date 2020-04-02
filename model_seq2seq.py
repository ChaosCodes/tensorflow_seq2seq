import tensorflow as tf
import numpy as np

class Seq2seq(object):
	
	def build_inputs(self, config):
		self.config = config
		self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
		self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
		self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
		self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')


	def encoder_layer(self, encoder_inputs, hidden_dim, dropout, cell_num=1, use_bi=True):
		# bi rnn
		if use_bi:
			((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=self.get_GRU_cell(hidden_dim, dropout), 
				cell_bw=self.get_GRU_cell(hidden_dim, dropout), 
				inputs=encoder_inputs, 
				sequence_length=self.seq_inputs_length, 
				dtype=tf.float32, 
				time_major=False
			)
			encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
			encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)
		else:
			encoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_GRU_cell(hidden_dim, dropout) for _ in range(cell_num)])
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)
		return encoder_outputs, encoder_state


	def get_GRU_cell(self, hidden_dim, dropout):
		return tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_dim), input_keep_prob=dropout)
	
	def make_attention_decoder(self, decoder_cell, encoder_outputs, encoder_state, useBeamSearch):
		if useBeamSearch > 1:
			tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=useBeamSearch)
			tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.seq_inputs_length, multiplier=useBeamSearch)
			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.config.hidden_dim, memory=tiled_encoder_outputs, memory_sequence_length=tiled_sequence_length)
			decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
			tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
			tiled_decoder_initial_state = decoder_cell.zero_state(batch_size=self.config.batch_size*useBeamSearch, dtype=tf.float32)
			tiled_decoder_initial_state = tiled_decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
			decoder_initial_state = tiled_decoder_initial_state
		else:
			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.seq_inputs_length)
			# attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.seq_inputs_length)
			decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
			decoder_initial_state = decoder_cell.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)
			decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
		return decoder_cell, decoder_initial_state
		
	
	def __init__(self, config, w2i_target, useAttention=True, useBeamSearch=1):
		initializer = np.random.random_sample((config.source_vocab_size, config.embedding_dim)) - 0.5
		self.word_init = initializer.astype(np.float32)

		self.build_inputs(config)
		l2_reg=tf.contrib.layers.l2_regularizer(0.05)
		with tf.variable_scope("encoder"):
		
			encoder_embedding = tf.get_variable('encoder_embedding', initializer=self.word_init, dtype=tf.float32, regularizer=l2_reg)
			encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)

			encoder_outputs, decoder_initial_state = self.encoder_layer(encoder_inputs_embedded, config.hidden_dim, config.dropout, use_bi=True)
		
		with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

			decoder_embedding = tf.get_variable('decoder_embedding', initializer=self.word_init, dtype=tf.float32, regularizer=l2_reg)
			tokens_go = tf.fill([config.batch_size], w2i_target["_GO"])

			decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1,1]), self.seq_targets[:,:-1]], 1)
			train_helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs), self.seq_targets_length)

			predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, w2i_target["_EOS"])

			with tf.variable_scope("gru_cell"):
				decoder_cell = self.get_GRU_cell(self.config.hidden_dim, self.config.dropout)
				if useAttention:
					decoder_cell, decoder_initial_state = self.make_attention_decoder(decoder_cell, encoder_outputs, decoder_initial_state, useBeamSearch)
				elif useBeamSearch > 1:
					decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)

		if useBeamSearch > 1:
			train_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, tokens_go, w2i_target["_EOS"],  decoder_initial_state, beam_width=useBeamSearch, output_layer=tf.layers.Dense(config.target_vocab_size))
			train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, maximum_iterations=tf.reduce_max(self.seq_targets_length))
			predict_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predict_helper, decoder_initial_state, output_layer=tf.layers.Dense(config.target_vocab_size))
			predict_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, maximum_iterations=tf.reduce_max(self.seq_targets_length))
		else:
			# run decoder and get the output
			train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, decoder_initial_state, output_layer=tf.layers.Dense(config.target_vocab_size))
			train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, maximum_iterations=tf.reduce_max(self.seq_targets_length))

			predict_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predict_helper, decoder_initial_state, output_layer=tf.layers.Dense(config.target_vocab_size))
			predict_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, maximum_iterations=tf.reduce_max(self.seq_targets_length))

		if useBeamSearch > 1:
			self.out = train_decoder_outputs.predicted_ids[:,:,0]
			self.predict_out = predict_decoder_outputs.predicted_ids[:,:,0]
		else:	
			# process the out and get the likely word id
			decoder_logits = train_decoder_outputs.rnn_output
			self.out = tf.argmax(decoder_logits, 2)

			predict_decoder_logits = predict_decoder_outputs.rnn_output
			self.predict_out = tf.argmax(predict_decoder_logits, 2)
			
			sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)

			self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.seq_targets, weights=sequence_mask)
			# self.predict_loss = tf.contrib.seq2seq.sequence_loss(logits=predict_decoder_logits, targets=self.seq_targets, weights=sequence_mask)
			# self.total_loss = self.loss + self.predict_loss

			self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
