
from glob import glob
import itertools
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import BeckerRNN.utils as utils
from BeckerRNN.network import Network

class FeatureExtractor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        files = glob(self.model_dir+'*.meta')
        self.model_name = max(files , key = os.path.getctime)
        self.model_name = self.model_name.rstrip('.meta')
        self.metadata_path = model_dir+'metadata_train.json'

    def get_features(self, tfrecords):
        net = Network()
        predictions = net.predict(tfrecords, self.model_name, self.metadata_path)
        ts_features = FeatureExtractor.preprocess_ts(predictions)
        return ts_features
        
    def get_time(ts_ids):
        obs_dir = '../../../../data/observations'
        dir_ = f'{obs_dir}/gaia/G/'
        ts_parameters = {'na_filter':False, 'usecols':['time', 'mag']}
        times = []
        for id_ in ts_ids:
            path = dir_ + str(id_) + '.dat'
            ts = pd.read_csv(path, **ts_parameters)
            min_time = ts['time'].min()
            max_time = ts['time'].max()
            times.append({'min_time':min_time, 'max_time':max_time, 'id':id_})
        
        times = pd.DataFrame(times)
        times = times.set_index('id')
        return times
        
    def preprocess_ts(ts_features):
        
        data = ts_features['all_h']
        dim0 = len(data) # Number of sources
        # dim1 = int(max(ts_features['lengths'])) # Max number of observations within a single lightcurve
        dim1 = len(max(data, key=len)) # Max number of observations within a single lightcurve
        dim2 = len(data[0][0]) # Hidden state dim

        data_ = np.full((dim0,dim1,dim2,), 0.0)
        for i, item in enumerate(data):
            data_[i,:len(item)] = item

        names = [f'h_{cell}_{h_index}' for cell, h_index in itertools.product(range(dim1),range(dim2))]
        ts_df = pd.DataFrame(data_.reshape(dim0,dim1*dim2), columns=names)
        
        scaler = MinMaxScaler()
        ts_df['id'] = ts_features['ids'].astype(int)
        ts_df = ts_df.add_suffix('_gaia')
        ts_df = ts_df.set_index('id_gaia', drop=True)
        ts_df[ts_df.columns] = scaler.fit_transform(ts_df)
        ts_df['lengths_gaia'] = [len(source) for source in data] #ts_features['lengths'].astype(int)
        ts_time = FeatureExtractor.get_time(ts_df.index)
        ts_df = ts_df.join(ts_time)
        
        return ts_df
        
if __name__=='__main__':
    ftrs_dir = '../../../../data/features'
    tfrecords = ['./Serialized/Xmatch.tfrecords']
    model_dir = './Runs/job1/Model/'
    save_path = f'{ftrs_dir}/gaia-sdss/features_rnn.csv'
    
    ts_model = FeatureExtractor(model_dir)
    ts_features = ts_model.get_features(tfrecords)
    ts_features.to_csv(save_path)