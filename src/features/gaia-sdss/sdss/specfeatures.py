import numpy as np
import os
import pandas as pd
import sys 
import tensorflow as tf
from specreader import SpecReader

class FeatureExtractor:
	def __init__(self, model_dir):
		self.model_dir = model_dir
		self.model_path = model_dir + 'models/model99.h5'

		params_path = model_dir + 'params.npy'
		config_path = model_dir + 'config.npy'
		self.params = np.load(params_path, allow_pickle=True).item()
		self.config = np.load(config_path, allow_pickle=True).item()

		self.import_scripts(model_dir)

	def import_scripts(self, model_dir):
		sys.path.append(model_dir)
		from model import Model, enc_dec_tensors
		self.model = Model
		self.enc_dec_tensors = enc_dec_tensors

	def build_db(self, data_dir, threshold):
		spec_reader = SpecReader()
		spec_reader.build_db(data_dir,
							 n_specs=-1,
							 normed=True,
							 down_sample=True,
							 threshold=threshold
							)

		# All wave lengths equal
		assert (spec_reader.wlen == spec_reader.wlen[0]).all()
		
		db = {'flux':spec_reader.flux, 
			  'wlen_same': spec_reader.wlen_same, 
			  'names': spec_reader.names
			 }
		return db

	def get_enc_dec(self, x, config, model_path):
		batch_size = x.shape[0]
		original_dim = config['original_dim']
		latent_dim = config['latent_dim']
		rate = config['rate']

		tf.reset_default_graph()
		with tf.Session() as sess:
			inputs, eps, _, _, _, _, _, _, _ = self.model(batch_size, original_dim, latent_dim, rate)
			enc_mean, dec_mean = self.enc_dec_tensors()

			saver = tf.train.Saver()
			saver.restore(sess, model_path)

			eps_in = np.full([x.shape[0], latent_dim], 0)
			feed_dict = {inputs: x, eps: eps_in}
			enc_x, dec_x = sess.run([enc_mean, dec_mean],feed_dict=feed_dict)
			
		return enc_x, dec_x

	def get_features(self, data_dir):
		w_min = self.params['w_min']
		w_max = self.params['w_max']
		threshold = [w_min, w_max]
		spectra = self.build_db(data_dir, threshold)
		X = spectra['flux']
		enc_x, _ = self.get_enc_dec(X, self.config, self.model_path)
		spectra['features'] = enc_x
		return spectra