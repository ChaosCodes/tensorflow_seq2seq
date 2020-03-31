import tensorflow as tf
import numpy as np
import random
import time
import os

from model_seq2seq_contrib import Seq2seq
# from model_seq2seq import Seq2seq

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True 
dataset_file = os.path.join(os.path.abspath('.'), 'dataset', 'COVID-Dialogue.txt')


class Config(object):
	embedding_dim = 128
	hidden_dim = 64
	batch_size = 64
	learning_rate = 0.005
	source_vocab_size = None
	target_vocab_size = None


def load_data(path):
	# num2en = {"1":"one", "2":"two", "3":"three", "4":"four", "5":"five", "6":"six", "7":"seven", "8":"eight", "9":"nine", "0":"zero"}
	docs_source = []
	docs_target = []
	# for i in range(10000):
	# 	doc_len = random.randint(1,8)
	# 	doc_source = []
	# 	doc_target = []
	# 	for j in range(doc_len):
	# 		num = str(random.randint(0,9))
	# 		doc_source.append(num)
	# 		doc_target.append(num2en[num])
	# 	docs_source.append(doc_source)
	# 	docs_target.append(doc_target)
	pair = []
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
	return docs_source, docs_target

	
def make_vocab(docs):
	w2i = {"_PAD":0, "_GO":1, "_EOS":2}
	i2w = {0:"_PAD", 1:"_GO", 2:"_EOS"}
	for doc in docs:
		for w in doc:
			if w not in w2i:
				i2w[len(w2i)] = w
				w2i[w] = len(w2i)
	return w2i, i2w
	
	
def doc_to_seq(docs):
	w2i = {"_PAD":0, "_GO":1, "_EOS":2}
	i2w = {0:"_PAD", 1:"_GO", 2:"_EOS"}
	seqs = []
	for doc in docs:
		seq = []
		for w in doc:
			if w not in w2i:
				i2w[len(w2i)] = w
				w2i[w] = len(w2i)
			seq.append(w2i[w])
		seqs.append(seq)
	return seqs, w2i, i2w


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size, batch_num):
	source_len = len(docs_source)

	source_batch = []
	target_batch = []
	
	source_lens = [len(docs_source[(batch_num + i) % source_len]) for i in range(batch_size)]
	target_lens = [len(docs_target[(batch_num + i) % source_len]) for i in range(batch_size)]
	
	max_source_len = max(source_lens)
	max_target_len = max(target_lens)
		
	for i in range(batch_size):
		source_seq = [w2i_source[w] for w in docs_source[(batch_num + i) % source_len]] +
						[w2i_source["_PAD"]]*(max_source_len-len(docs_source[(batch_num + i) % source_len]))
		target_seq = [w2i_target[w] for w in docs_target[(batch_num + i) % source_len]] + [w2i_target["_EOS"]] +
						[w2i_target["_PAD"]]*(max_target_len-1-len(docs_target[(batch_num + i) % source_len]))
		source_batch.append(source_seq)
		target_batch.append(target_seq)
	
	return source_batch, source_lens, target_batch, target_lens, (batch_num + batch_size) % batch_size
	
	
if __name__ == "__main__":

	print("(1)load data......")
	docs_source, docs_target = load_data(dataset_file)
	w2i_source, i2w_source = make_vocab(docs_source)
	w2i_target, i2w_target = make_vocab(docs_target)
	
	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(w2i_source)
	config.target_vocab_size = len(w2i_target)
	model = Seq2seq(config=config, w2i_target=w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1)
	
	
	print("(3) run model......")
	batches = 30000
	print_every = 100
	batch_num = 0
	epoch = 20
	with tf.Session(config=tf_config) as sess:
		tf.summary.FileWriter('graph', sess.graph)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		
		losses = []
		total_loss = 0
		for _ in range(epoch):
			for batch in range(batches):
				source_batch, source_lens, target_batch, target_lens, batch_num = get_batch(docs_source, w2i_source, docs_target, w2i_target, config.batch_size, batch_num)
				
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
					predict_batch = sess.run(model.out, feed_dict)
					for i in range(3):
						print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"])
						print("out:",[i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"])
						print("tar:",[i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"])
						print("")
			
			print(losses)
		print(saver.save(sess, "checkpoint/model.ckpt"))		
		
	


