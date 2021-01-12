from astropy.io import fits
import os
import joblib
from joblib import Parallel, delayed
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import time
from tqdm import tqdm

scaler = MinMaxScaler()
n_jobs = 5 

def get_spectra(spec_path, normed):
    hdulist = fits.open(spec_path)
    flux = np.array(hdulist[1].data["flux"])
    loglam = np.array(hdulist[1].data["loglam"])
    wlen = np.array([10**(w) for w in loglam])
    if normed:
        flux = flux.reshape(-1,1)
        scaler.fit(flux)
        flux = scaler.transform(flux) 
        flux = flux.reshape(-1)
    return wlen, flux

def down_sample(flux,wlen,keep):
    return flux[keep], wlen[keep]

class SpecReader:
    def __init__(self, root='./'):
        self.wlen = []
        self.flux = []
        self.names = []
        self.wlen_same = []
        self.w_min = None
        self.w_max = None
        self.root = root

    def down_sample_all(self, threshold):

        mins = [w[0] for w in self.wlen]
        maxs = [w[-1] for w in self.wlen]
        num_stars = len(self.wlen)
        
        if not (threshold is None):
            self.w_min = threshold[0]
            self.w_max = threshold[1]
        else:
            # Delete 1%
            offset = int(len(mins)*0.01)
            self.w_min = sorted(mins)[num_stars-1-offset]
            self.w_max = sorted(maxs)[offset]
            borders = {'mins':mins, 'maxs':maxs, 'w_min':self.w_min, 'w_max':self.w_max}
            np.save(self.root+'params.npy', borders)
        
        keep = (mins <= self.w_min) & (maxs >= self.w_max)
        self.flux = self.flux[keep]
        self.wlen = self.wlen[keep]
        self.names = self.names[keep]

        w_keep = [(w>=self.w_min) & (w<=self.w_max) for w in tqdm(self.wlen)]
        self.wlen_same = self.wlen[0][w_keep[0]]

        down_sampled = Parallel(n_jobs=n_jobs)(delayed(down_sample)(self.flux[i],self.wlen[i],w_keep[i]) for i in tqdm(range(len(self.flux))))
        down_sampled = np.array(down_sampled)
        self.flux = np.array(down_sampled[:,0])
        self.wlen = np.array(down_sampled[:,1])

    def build_db(self, spec_dir, n_specs=-1, normed=False, down_sample=False, threshold=None):
        files = list(filter(lambda x: '.fits' in x, os.listdir(spec_dir)))
        if n_specs == -1:
            self.names = np.array(files)
        else:
            self.names = np.array(files[:n_specs])

        s = time()
        specs = Parallel(n_jobs=n_jobs)(delayed(get_spectra)(spec_dir+name, normed) for name in tqdm(self.names))
        specs = np.array(specs)
        self.wlen = specs[:,0]
        self.flux = specs[:,1]
        e = time()
        print("Time to read: ", e-s)

        if down_sample:
            s = time()
            self.down_sample_all(threshold)
            e = time()
            print("Time to down sample: ",e-s)