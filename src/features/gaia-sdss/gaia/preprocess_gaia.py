import sys
sys.path.append('./BeckerRNN')
import os
import preprocesser as p

ctlg_dir = '../../../../data/catalogues'
data_save = './Serialized/'
file_train = f'{ctlg_dir}/gaia/gaia-rnn.dat'

min_L = 500
max_L = 10000
min_N = 15
max_N = 2000
w = 4
s = 2
time = False

lc_parameters = {'header':0, 'na_filter':False,'sep':',','usecols':['time', 'mag', 'mag_err']}
num_cores = 4

preprocesser = p.Preprocesser(max_L=max_L, min_L=min_L, min_N=min_N, max_N=max_N, w=w, s=s,
               num_cores=num_cores, lc_parameters=lc_parameters,w_time = time)
preprocesser.prepare(file_train=file_train, save_dir=data_save)