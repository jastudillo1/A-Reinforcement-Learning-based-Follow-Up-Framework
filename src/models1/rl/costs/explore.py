import itertools
from joblib import Parallel, delayed
import json
import numpy as np
import os
import pandas as pd
import pickle
from time import time
from tqdm import tqdm

from strategies import strategies, compare, metrics
from trees import *

def make_settings():
    cost_spec = list(range(1,10))[::2]
    cost_coeff = np.linspace(1/40,1/10,4)

    run_path = '{}/runs.json'.format(save_dir)
    runs = {}
    for i, args in enumerate(itertools.product(cost_spec, cost_coeff)):
        runs[i] = {'cost_spec': args[0], 'cost_coeff': args[1]}

    with open(run_path, 'w') as f:
        json.dump(runs, f)

    return runs

def save_run_settings(root_args, tree_args, save_path):
    data = {}
    clf = tree_args.pop('all_clf')
    data['tree_args'] = tree_args
    data['root_args'] = root_args
    
    with open(save_path, 'w') as f:
        json.dump(data, f)
        
    tree_args['all_clf'] = clf
    
def strat_summ_fn(root_args, tree_args, save_dir, run_id, features, rl_model=None, processor=None):
    
    comparison_path = '{}/comparison{}.csv'.format(save_dir,run_id)
    cost_path = '{}/cost{}.csv'.format(save_dir,run_id)
    reward_path = '{}/reward{}.csv'.format(save_dir,run_id)
    prob_path = '{}/prob{}.csv'.format(save_dir,run_id)
    settings_path = '{}/settings{}.json'.format(save_dir,run_id)
    summary_path = '{}/summary{}.csv'.format(save_dir,run_id)
    
    range_ = range(features.shape[0])
    
    strat_summ = []
    # for i in tqdm(range_):
    #     strat_summ_i = strategies(features.iloc[i], root_args, tree_args, rl_model, processor)
    #     strat_summ.append(strat_summ_i)
    strat_summ = Parallel(n_jobs=-1)(delayed(strategies)(features.iloc[i], root_args, tree_args, rl_model, processor)
                            for i in tqdm(range_))

    strat_summ = pd.DataFrame(list(itertools.chain(*strat_summ)))
    strat_summ.to_csv(summary_path, index=True)

    strat_keys=['prob_greedy', 'reward_greedy', 'photometry_only']
    if not rl_model is None:
        strat_keys.append('RL')
    
    strat_comp = compare(strat_summ, strat_keys)
    strat_comp.to_csv(comparison_path, index=True)
    
    strat_reward = metrics(strat_comp, 'reward', strat_keys)
    strat_reward.to_csv(reward_path, index=True)
    
    strat_cost = metrics(strat_comp, 'cost', strat_keys)
    strat_cost.to_csv(cost_path, index=True)

    strat_prob = metrics(strat_comp, 'probability', strat_keys)
    strat_prob.to_csv(prob_path, index=True)
    
    save_run_settings(root_args, tree_args, settings_path)

def run(features, clf_df, settings, id_, rl_model=None, processor=None):

    print(id(clf_df))

    root_args = {'n_obs': 5, 
                 'n_spec': 0, 
                 'n_color': 0
                }

    tree_args = {'cost_obs': 1,
                 'cost_spec': settings['cost_spec'], 
                 'cost_col': 1, 
                 'cost_coeff': settings['cost_coeff'],
                 'all_clf': clf_df,
                 'p_thres': None
                }

    start = time()
    strat_summ_fn(root_args, tree_args, save_dir, id_, features, rl_model, processor)
    end = time()
    msg = 'Total time: {:.2f} s.'.format(end-start)

if __name__ == '__main__':
    
    # Local paths
    # ftrs_dir = '../../../../data/features'
    # features_path = f'{ftrs_dir}/gaia-sdss/features.csv'
    # splits_path = '../../splits.pk'
    # clf_path =  '../../rf/classifiers.pkl'
    # save_dir = './strategies'
    
    # Cluster paths
    features_path = 'job_data/features.csv'
    splits_path = 'job_data/splits.pk'
    clf_path =  'job_data/classifiers.pkl'
    save_dir = './strategies'
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    features = pd.read_csv(features_path)
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    
    rl_train = features.set_index('id_gaia').loc[splits['rl_train']].reset_index()
    clf_df = pd.read_pickle(clf_path)

    runs = make_settings()
    strat_summ = Parallel(n_jobs=-1)(delayed(run)(rl_train, clf_df, settings, id_)
                        for id_, settings in runs.items())
    # for run_id, settings in runs.items():
    #     run(test_features, clf_df, settings, run_id)
        # break