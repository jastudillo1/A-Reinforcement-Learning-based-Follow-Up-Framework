import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def include_splits(features, splits, drop_rows=True):
    features.set_index('id_gaia', inplace=True, drop=True)
    features['split'] = None

    for split_label, split_ids_path in splits.items():
        ids_ = pd.read_csv(split_ids_path).values.reshape(-1)
        features.loc[ids_,'split'] = split_label
    if drop_rows:
        features.dropna(subset=['split'],inplace=True)
    features.reset_index(inplace=True, drop=False)

    return features

def make_splits(features, clf_size, rl_size, test_size, save_dir, filter_cls=None):
    
    assert (clf_size + rl_size + test_size) == 1

    clf_save = save_dir + '/clf_ids.csv'
    rl_save = save_dir + '/rl_ids.csv'
    test_save = save_dir + '/test_ids.csv'

    if not filter_cls is None:
        features = features[features['label'].isin(filter_cls)]
        features.reset_index(inplace=True, drop=True)

    index = range(features.shape[0])
    labels = features['label']
    rl_size_norm = rl_size/(clf_size+rl_size)
    clf_index, test_index = train_test_split(index, stratify=labels, test_size=test_size, random_state=0)
    clf_index, rl_index = train_test_split(clf_index, stratify=labels[clf_index], test_size=rl_size_norm, random_state=0)

    clf_set = set(clf_index)
    rl_set = set(rl_index)
    test_set = set(test_index)

    assert len(clf_set.intersection(rl_set)) == 0
    assert len(clf_set.intersection(test_set)) == 0
    assert len(rl_set.intersection(test_set)) == 0
    assert len(clf_set.union(rl_set).union(test_set)) == features.shape[0]

    clf_ids = features.id_gaia.iloc[clf_index]
    rl_ids = features.id_gaia.iloc[rl_index]
    test_ids = features.id_gaia.iloc[test_index]

    clf_ids.to_csv(clf_save, index=False, header=True)
    rl_ids.to_csv(rl_save, index=False, header=True)
    test_ids.to_csv(test_save, index=False, header=True)

def ts_features_row(row):
    '''
    Get photometric features columns (location).
    Last hidden state for each sample in dataset ([h_[sample_N_obs]_[0...features_dim]])
    '''
    prefix = 'h_'+str(row.lengths_gaia-1)+'_'
    features_cols = [c for c in row.index.values if prefix in c]
    return row[features_cols].values

def ts_features(features):
    ts_features = np.vstack(features.apply(ts_features_row, axis=1))

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

def clf_df_train(clf_df, datasets_train, datasets_test):
    
    y_train = datasets_train['labels']
    y_test = datasets_test['labels']
    
    for clf_index, row in clf_df.iterrows():
        src_include = row[row==1]
        src_include = src_include.index 
        X_train = build_X(datasets_train, src_include)
        X_test = build_X(datasets_test, src_include)
        clf = train_rf(X_train, y_train)
        clf_df.at[clf_index,'clf'] = clf
        
        descr_base = ', '.join(src_include)
        classes = clf.classes_

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        descr = descr_base + '\nTrain set'
        plot_cm(y_train, y_train_pred, classes, descr)
        print(classification_report(y_train, y_train_pred))

        descr = descr_base + '\nTest set'
        plot_cm(y_test, y_test_pred, classes, descr)
        print(classification_report(y_test, y_test_pred))
    
    return clf_df

def train_clf(features_train, features_test, save_dir):

    clf_save = save_dir + '/classifiers.pkl'

    datasets_train = build_sets(features_train)
    datasets_test = build_sets(features_test)
    clf_df = clf_df_base()
    clf_df = clf_df_train(clf_df, datasets_train, datasets_test)
    clf_df.to_pickle(clf_save)
    

def clf_df_plots(clf_df, features):
    
    datasets = build_sets(features)
    y = features.label
    
    for i, item in clf_df.iterrows():
        src_include = item[item==1]
        src_include = src_include.index
        X = build_X(datasets, src_include)
        clf = clf_df.loc[i]['clf']
        
        y_pred = clf.predict(X)
        classes = clf.classes_
        descr = ', '.join(src_include)
        plot_cm(y, y_pred, classes, descr)
        print('Accuracy: {:.2f}'.format(accuracy_score(y, y_pred)))
        print(classification_report(y, y_pred))
        
def plot_cm_ax(ax, cm, classes, normalize, title):
    
    ax.set_title(title)

    labels = classes#[self.trans[i] for i in range(self.num_classes)]
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm[i,j] ='%.2f' %cm[i,j]

    thresh = 0.001
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] < thresh else 'black')

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def plot_cm(y, y_pred, classes, descr):

    plt.clf()
    plt.rc('font', size=15)
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.rc('figure', titlesize=22)

    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,12))
    cm = confusion_matrix(y, y_pred)
    title = 'Confusion Matrix ' + descr + ' features'
    plot_cm_ax(ax0, cm, normalize=False, classes=classes, title='Normalized '+ title)
    plot_cm_ax(ax1, cm, normalize=True, classes=classes, title=title)

    plt.tight_layout()
    plt.show()

