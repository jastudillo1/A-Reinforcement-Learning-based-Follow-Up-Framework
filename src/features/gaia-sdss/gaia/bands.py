from astropy.io.votable import parse_single_table
from collections import Counter
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd

def read_file(file):
    votable = parse_single_table(file)
    table = votable.to_table()
    df = pd.DataFrame(np.array(table))
    return df

def to_files(save_dir, df, id_):
    # Save metadata
    meta = {}
    
    # Select one source_id
    df = df[df['source_id']==id_]
    df = df[['band','time','mag','mag_err','flux','flux_error']]
    
    # Get each band in objects separately
    df['band'] = df['band'].str.decode(encoding='utf-8')
    df_bands = [[key, df.drop('band', axis=1)] for key, df in df.groupby('band')]
    df_bands = dict(df_bands)
    
    # Create directories
    for band in df_bands.keys():
        band_dir = save_dir+'/'+band
        if not os.path.exists(band_dir):
            os.mkdir(band_dir)
    
    # Save into separated files
    for band, df in df_bands.items(): 
        path = '{}/{}/{}.dat'.format(save_dir, band, id_)
        df.to_csv(path, index=False,index_label=False, float_format='%.8f')
        meta[band] = {'ID': id_, 'Address': path, 'N': df_bands[band].shape[0]}
        
    return meta

def split_bands(read_dir, save_dir):
    files = glob(read_dir)
    meta = {}
    
    for file in files:
        df = read_file(file)
        
        # Create error in magnitude
        err = 2.5*np.power(np.log10(np.e),2)*df.flux_error/df.flux
        df = df.assign(mag_err = err)

        # Remove unnecesary columns
        df = df.drop(['solution_id','transit_id','flux_over_error','other_flags'],axis=1)

        # Inverse reyected obs to filter
        df['rejected_by_photometry'] = np.logical_not(df['rejected_by_photometry'])
        df['rejected_by_variability'] = np.logical_not(df['rejected_by_variability'])

        # Filter observations
        df = df[df['rejected_by_photometry']]
        df = df[df['rejected_by_variability']]

        # Remove boolean filters
        df = df.drop(['rejected_by_photometry','rejected_by_variability'],axis=1)

        # Get unique sources
        sources = list(set(df['source_id']))
        if len(sources)>1:
            raise ValueError('More than one source in file: ', sources)
        
        # Split and save
        meta_i = to_files(save_dir, df, sources[0])
#         print(meta_i)
        
        for band, meta_i_band in meta_i.items():
            if band in meta:
                meta[band].append(meta_i_band)
            else:
                meta[band] = [meta_i_band]

    meta = [[key, pd.DataFrame.from_dict(mband)] for key, mband in meta.items()]
    meta = dict(meta)
    return meta
    
def save_meta(meta, labels_path):
    labels = pd.read_csv(labels_path)
    labels = labels.set_index('source_id_gaia')
    labels = labels.rename(columns={'best_class_name_gaia': 'Class'})
    labels = labels['Class']

    for band, meta_b in meta.items():
        path = './Gaia-{}.dat'.format(band)
        meta_b = meta_b.set_index('ID')
        meta_b = meta_b.join(labels).reset_index()
        meta_b = meta_b[['ID','Class','Address','N']]
        meta_b.to_csv(path, index=False)
        
if __name__=='__main__':

    ctlg_dir = '../../../../data/catalogues'
    obs_dir = '../../../../data/observations'
    read_dir = f'{obs_dir}/gaia/xmatch/all-bands/*.vot'
    save_dir = f'{obs_dir}/gaia/xmatch/bands'
    labels_path = f'{ctlg_dir}/gaia-sdss/cross-match-labels.csv'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    meta = split_bands(read_dir, save_dir)
save_meta(meta, labels_path)