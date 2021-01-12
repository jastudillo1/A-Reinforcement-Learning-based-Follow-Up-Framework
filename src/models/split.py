# Split cross-match dataset into three main sets:
# (1) Data for trainings Classifiers
# (2) Data for training Reinforcement Learning Framework
# (3) Data for testing Reinforcement Learning Framework
# We split them so as to have out of sample classification results when testing Reinforcement Learning Framework.

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def make_splits(features, clf_size, rl_size, test_size, save_dir, min_support=0):
    
    assert (clf_size + rl_size + test_size) == 1 # Splits add up to 1
    assert (min_support<=1) & (min_support>=0) # Valid class support
    assert len(set(features.id_gaia))==features.shape[0] # No repetitions
    
    save_path = f'{save_dir}/splits.pk'

    cls_, counts = np.unique(features.label, return_counts=True)
    perc = counts/sum(counts)
    cls_keep = cls_[perc>=min_support]
    features = features[features['label'].isin(cls_keep)]
    features.reset_index(inplace=True, drop=True)

    index = range(features.shape[0])
    labels = features['label']
    rl_size_norm = rl_size/(clf_size+rl_size)
    clf_index, test_index = train_test_split(index, stratify=labels, test_size=test_size, random_state=0)
    clf_index, rl_index = train_test_split(clf_index, stratify=labels[clf_index], test_size=rl_size_norm, random_state=0)
    
    splits = {
        'clf_train': features.iloc[clf_index].id_gaia.values,
        'rl_train': features.iloc[rl_index].id_gaia.values,
        'test': features.iloc[test_index].id_gaia.values,
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    
if __name__=='__main__':
    ftrs_dir = '../../data/features'
    features_path = f'{ftrs_dir}/gaia-sdss/features.csv'
    save_dir = '.'

    clf_size = 0.4
    rl_size = 0.4
    test_size = 0.2
    features = pd.read_csv(features_path)
    
    make_splits(features, clf_size, rl_size, test_size, save_dir, min_support=0.01)