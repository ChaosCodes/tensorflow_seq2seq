import tensorflow as tf
import numpy as np
import random
import time
import os

from model_seq2seq_contrib import Seq2seq
from utils import ensure_dir, make_store_path
# from model_seq2seq import Seq2seq

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True 
dataset_file = os.path.join(os.path.abspath('.'), 'dataset', 'COVID-Dialogue.txt')

# set the gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class Config(object):
	embedding_dim = 100
	hidden_dim = 400
	batch_size = 32
	learning_rate = 0.0005
	source_vocab_size = None
	target_vocab_size = None


def load_data(path):
	docs_source = []
	docs_target = []

	with open(path, 'r') as f:
		current_sentence = ''
		last_sentence = ''
		while True:
			line = f.readline()
			if not line:
				break
			if line[:3] == 'id=':
				current_sentence = ''
				last_sentence = ''
			else:
				current_sentence = line[2:]
				if last_sentence != '':
    				
					docs_source.append(last_sentence.split())
					docs_target.append(current_sentence.split())
				last_sentence = current_sentence
		idx = list(range(len(docs_source)))
		random.shuffle(idx)
		train_idx = idx[:int(len(idx) * 0.8)]
		val_idx = idx[int(len(idx) * 0.8): int(len(idx) * 0.9)]
		test_idx = idx[int(len(idx) * 0.9):]

		docs_source = np.array(docs_source)
		docs_target = np.array(docs_target)

		train_source = docs_source[train_idx]
		train_target = docs_target[train_idx]
		val_source = docs_source[val_idx]
		val_target = docs_target[val_idx]
		test_source = docs_source[test_idx]
		test_target = docs_target[test_idx]

	return docs_source, docs_target, train_source, train_target, val_source, val_target, test_source, test_target

	
def make_vocab(docs):
	w2i = {"_PAD":0, "_GO":1, "_EOS":2}
	i2w = {0:"_PAD", 1:"_GO", 2:"_EOS"}
	for doc in docs:
		for w in doc:
			word = w.lower()
			if word not in w2i:
				i2w[len(w2i)] = word
				w2i[word] = len(w2i)
	return w2i, i2w


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size, batch_num):
	ps = list(range(batch_num, batch_num + batch_size))

	source_batch = []
	target_batch = []

	docs_source_len = len(docs_source)
	docs_target_len = len(docs_target)
	source_lens = [len(docs_source[p % docs_source_len]) for p in ps]
	target_lens = [len(docs_target[p % docs_target_len])+1 for p in ps]

	max_source_len = max(source_lens)
	max_target_len = max(target_lens)


	for p in ps:
		source_seq = [w2i_source[w.lower()] for w in docs_source[p % docs_source_len]] + [w2i_source["_PAD"]]*(max_source_len-len(docs_source[p % docs_source_len]))			
		target_seq = [w2i_target[w.lower()] for w in docs_target[p % docs_target_len]] + [w2i_target["_EOS"]] + [w2i_target["_PAD"]]*(max_target_len-1-len(docs_target[p % docs_target_len]))

		source_batch.append(source_seq)
		target_batch.append(target_seq)


	return source_batch, source_lens, target_batch, target_lens, (batch_num + batch_size) % docs_source_len
	
	
if __name__ == "__main__":

	print("(1)load data......")
	docs_source, docs_target, train_source, train_target, val_source, val_target, test_source, test_target = load_data(dataset_file)
	w2i_source, i2w_source = make_vocab(docs_source)
	w2i_target, i2w_target = make_vocab(docs_target)
	
	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(w2i_source)
	config.target_vocab_size = len(w2i_target)
	model = Seq2seq(config=config, w2i_target=w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1)
	
	
	print("(3) run model......")
	batches = 3000
	print_every = 100
	batch_num = 0
	val_batch_num = 0
	test_batch_num = 0
	epoch = 200

	load = True
	with tf.Session(config=tf_config) as sess:
		# tf.summary.FileWriter('graph', sess.graph)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		store_dir = make_store_path()
		ensure_dir(store_dir)
		if load:
			print('load the model')
			ckpt = tf.train.get_checkpoint_state('save_model/' + make_store_path)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('load the pretrained model parameters')
			else:
				print('not loading')
				sess.run(tf.global_variables_initializer())
		else:
			print('not loading')
			sess.run(tf.global_variables_initializer())
		
		losses = []
		total_loss = 0
		for e in range(epoch):
			for batch in range(batches):
				source_batch, source_lens, target_batch, target_lens, batch_num = get_batch(train_source, w2i_source, train_target, w2i_target, config.batch_size, batch_num)
				
				feed_dict = {
					model.seq_inputs: source_batch,
					model.seq_inputs_length: source_lens,
					model.seq_targets: target_batch,
					model.seq_targets_length: target_lens
				}
				
				loss, _ = sess.run([model.loss, model.train_op], feed_dict)
				total_loss += loss
				
				if batch % print_every == 0 and batch > 0:

					print_loss = total_loss if batch == 0 else total_loss / print_every
					losses.append(print_loss)
					total_loss = 0
					print("-----------------------------")
					print("batch:",batch,"/",batches)
					print("time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
					print("loss:",print_loss)
					
					print("samples:\n")
					source_batch, source_lens, target_batch, target_lens, val_batch_num = get_batch(val_source, w2i_source, val_target, w2i_target, config.batch_size, val_batch_num)
					val_feed_dict = {
						model.seq_inputs: source_batch,
						model.seq_inputs_length: source_lens,
						model.seq_targets: target_batch,
						model.seq_targets_length: target_lens
					}
					predict_batch, val_loss = sess.run([model.out, model.loss], feed_dict)
					print("loss:", val_loss)

					for i in range(3):
						print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"])
						print("out:",[i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"])
						print("tar:",[i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"])
						print("")
			### test
			print(f"-------epoch {e}-----test--------------")
			source_batch, source_lens, target_batch, target_lens, test_batch_num = get_batch(test_source, w2i_source, test_target, w2i_target, config.batch_size, test_batch_num)
			val_feed_dict = {
				model.seq_inputs: source_batch,
				model.seq_inputs_length: source_lens,
				model.seq_targets: target_batch,
				model.seq_targets_length: target_lens
			}
			predict_batch, test_loss = sess.run([model.out, model.loss], feed_dict)
			print("loss:", test_loss)

			for i in range(3):
				print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"])
				print("out:",[i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"])
				print("tar:",[i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"])
				print("")

			saver.save(sess, os.path.join(os.path.abspath('.'), 'save_model', store_dir))
