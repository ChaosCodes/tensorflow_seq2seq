import tensorflow as tf
import numpy as np
import random
import os
import time
from model_seq2seq import Seq2seq
from train_seq2seq import load_data, make_vocab, get_batch
from train_seq2seq import Config
from config import load_arguments
from utils import make_store_path
# from model_seq2seq import Seq2seq

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True 


if __name__ == "__main__":
	print("(1)load data......")
	args = load_arguments()
	# set the gpu_id
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
	# set file
	dataset_file = os.path.join(os.path.abspath('.'), 'dataset', 'COVID-brief-Dialogue.txt' if args.breif == 0 else 'COVID-Dialogue.txt')

	docs_source, docs_target, _, _, _, _, _, _ = load_data(dataset_file, is_shuffle = False)
	w2i_source, i2w_source = make_vocab(docs_source)
	w2i_target, i2w_target = make_vocab(docs_target)
	
	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(w2i_source)
	config.target_vocab_size = len(w2i_target)
	model = Seq2seq(config=config, w2i_target=w2i_target, useTeacherForcing=False, useAttention=True, useBeamSearch=3)
	
	
	print("(3) run model......")
	# max_target_len = 20
	store_dir = make_store_path(config)
	store_path = os.path.join(os.path.abspath('.'), 'save_model', store_dir, 'seq2seq')
	
	with tf.Session(config=tf_config) as sess:
		saver = tf.train.Saver()
		saver.restore(sess, store_path)
		num = 20
		source_batch, source_lens, target_batch, target_lens, num = get_batch(docs_source, w2i_source, docs_target, w2i_target, config.batch_size, num)
		max_target_len = max(target_lens)
		feed_dict = {
			model.seq_inputs: source_batch,
			model.seq_inputs_length: source_lens,
			model.seq_targets: [[0]*max_target_len]*len(source_batch),
			model.seq_targets_length: [max_target_len]*len(source_batch)
		}
		
		print("samples:\n")
		predict_batch = sess.run(model.out, feed_dict)
		for i in range(len(source_batch)):
			print("in:", ' '.join([i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"]))
			print("out:", ' '.join([i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"]))
			print("tar:", ' '.join([i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"]))
			print("")
			
			