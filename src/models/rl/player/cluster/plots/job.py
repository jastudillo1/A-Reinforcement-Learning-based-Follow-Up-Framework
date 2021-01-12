from framework import *
from joblib import Parallel, delayed

def train_run(run):
    data_dir = './data/'
    features_path = data_dir + 'features-2.csv'
    clf_path =  data_dir + 'classifiers.pkl'
    rl_ids_path = data_dir + 'rl_ids.csv'
    test_ids_path = data_dir + 'test_ids.csv'
    settings_path = data_dir + 'settings8.json'
    reward_path = data_dir + 'exhaustive_info.csv'
    
    setup = SetUp(features_path, clf_path, rl_ids_path, test_ids_path, settings_path, reward_path)
    trainer_args = setup.trainer_args()
    trainer_args['run'] = run
    trainer = RLTrainer(**trainer_args)
    trainer.train()

if __name__ == '__main__':
    trainings = Parallel(n_jobs=2)(delayed(train_run)(run) for run in range(2))
