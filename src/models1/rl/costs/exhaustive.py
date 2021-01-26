# Save best path information for each source using the selected observational
# costs.

import json
import numpy as np
import pandas as pd
import pickle
from trees import BruteForceTree
from tqdm import tqdm

def get_exhaustive_info(source_ftrs, settings):
    
    settings['tree_args'].update({'all_features': source_ftrs})
    
    brute = BruteForceTree(**settings['tree_args'], strategy_name='brute_force')
    brute.create_root(**settings['root_args'])
    brute.build_tree()
    
    info = brute.strategy_info()
    info.update(brute.best_path[-1].settings)
    
    return info

if __name__=='__main__':
    ftrs_dir = '../../../../data/features'
    clf_path =  '../../rf/classifiers.pkl'
    features_path = f'{ftrs_dir}/gaia-sdss/features.csv'
    settings_path = 'strategies/settings8.json'
    splits_path = '../../splits.pk'
    save_path = './exhaustive.csv'
    
    with open(settings_path) as f:
        settings = json.load(f)
    
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    filter_ids = np.concatenate(list(splits.values()))
    features = pd.read_csv(features_path)
    features = features.set_index('id_gaia').loc[filter_ids].reset_index()
        
    clf_df = pd.read_pickle(clf_path)
    
    settings['tree_args']['all_features'] = None
    settings['tree_args']['all_clf'] = clf_df
    
    infos = [get_exhaustive_info(source_ftrs, settings) for _, source_ftrs in tqdm(features.iterrows())]
    infos = pd.DataFrame(infos)
    infos.to_csv(save_path, index=False)
    
    
    