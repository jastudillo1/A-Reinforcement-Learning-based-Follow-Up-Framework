import itertools
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import classification_report  

from plots import plot_cm

def ts_func(row):
    # Get features from the hidden state corresponding to the last observation per light curve.
    last_i = row.lengths_gaia-1
    last_cols = row.index[[f'h_{last_i}_' in c for c in row.index]]
    last_h = row[last_cols].values
    return last_h

def ts_features(features):
    ts_features = features.apply(ts_func, axis=1)
    ts_features = np.vstack(ts_features)

    return ts_features

def spec_features(features):
    sp_cols = features.columns[[('_sdss' in c) & (not 'id' in c) for c in features.columns]]
    sp_features = features[sp_cols].values

    return sp_features

def color_features(features):
    c_cols = ['color_gaia']
    c_features = features[c_cols].values

    return c_features

def build_sets(features):
    '''Build features sets for each different observational source: ['photo', 'spec', 'color']'''

    ts_features_ = ts_features(features)
    spec_features_ = spec_features(features)
    color_features_ = color_features(features)
    labels = features.label.values

    datasets = {'photo': ts_features_, 
                'spec':spec_features_, 
                'color': color_features_, 
                'labels':labels
                }
    return datasets

def train_rf(X, labels):
    n_estimators = range(5,40)[::2]
    max_depth = range(3,20)#ts_features.shape[-1])
    results = []

    search = itertools.product(n_estimators, max_depth)
    for n_estimators_, max_depth_ in search:
        clf = RandomForestClassifier(max_depth=max_depth_, n_estimators=n_estimators_, oob_score=True)
        clf.fit(X, labels)
        score = clf.oob_score_
        results.append({'n_estimators': n_estimators_, 'max_depth' : max_depth_, 'oob_score': score})
    
    results = pd.DataFrame(results)
    index = results.oob_score.idxmax()
    n_estimators = results.loc[index].n_estimators.astype(int)
    max_depth = results.loc[index].max_depth.astype(int)
    clf = RandomForestClassifier(max_depth=max_depth_, n_estimators=n_estimators).fit(X, labels)
    
    return clf

def build_X(datasets, src_include):
    X = [datasets[s] for s in src_include]
    X = np.hstack(X)
    return X

def clf_df_base(sources=['photo', 'spec', 'color']):
    options = list(itertools.product([0, 1], repeat=len(sources)))
    clf_all = pd.DataFrame(options, columns=sources)
    clf_drop = clf_all.sum(axis=1)==0
    clf_all = clf_all[~clf_drop]
    clf_all['clf'] = None
    return clf_all

def clf_df_train(clf_df, datasets_train, datasets_test, save_dir):
    
    report_path = f'{save_dir}/results.pk'
    plot_dir = f'{save_dir}/plots'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    
    y_train = datasets_train['labels']
    y_test = datasets_test['labels']
    report = {}
    
    for clf_index, row in clf_df.iterrows():
        src_include = row[row==1]
        src_include = src_include.index 
        X_train = build_X(datasets_train, src_include)
        X_test = build_X(datasets_test, src_include)
        clf = train_rf(X_train, y_train)
        clf_df.at[clf_index,'clf'] = clf
        
        classes = clf.classes_

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        descr = ', '.join(src_include) + '\nTrain set'
        descr_save = '_'.join(src_include) + '_train'
        plot_cm(y_train, y_train_pred, classes, descr, descr_save, plot_dir)
        report[descr_save] = classification_report(y_train, y_train_pred)

        descr = ', '.join(src_include) + '\nTest set'
        descr_save = '_'.join(src_include) + '_test'
        plot_cm(y_test, y_test_pred, classes, descr, descr_save, plot_dir)
        report[descr_save] = classification_report(y_test, y_test_pred)
        
    with open(report_path, 'wb') as f:
        pickle.dump(report, f)
    
    return clf_df

def train_clf(features_train, features_test, save_dir):

    clf_save = save_dir + '/classifiers.pkl'

    datasets_train = build_sets(features_train)
    datasets_test = build_sets(features_test)
    clf_df = clf_df_base()
    clf_df = clf_df_train(clf_df, datasets_train, datasets_test, save_dir)
    clf_df.to_pickle(clf_save)

if __name__=='__main__':
    ftrs_dir = '../../../data/features'
    features_path = f'{ftrs_dir}/gaia-sdss/features.csv'
    splits_path = '../splits.pk'
    save_dir = './'
    
    features = pd.read_csv(features_path)
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)    
    
    train = features.set_index('id_gaia').loc[splits['clf_train']].reset_index()
    test = features.set_index('id_gaia').loc[splits['test']].reset_index()
    train_clf(train, test, save_dir)