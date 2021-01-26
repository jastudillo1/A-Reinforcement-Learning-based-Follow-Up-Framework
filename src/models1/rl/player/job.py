from framework import *
from joblib import Parallel, delayed

def train_run(run):
    data_dir = './job_data'
    features_path = f'{data_dir}/features.csv'
    clf_path =  f'{data_dir}/classifiers.pkl'
    settings_path = f'{data_dir}/settings8.json'
    reward_path = f'{data_dir}/exhaustive.csv'
    splits_path = f'{data_dir}/splits.pk'
    
    setup = SetUp(features_path, clf_path, splits_path, settings_path, reward_path)
    trainer_args = setup.trainer_args()
    trainer_args['run'] = run
    trainer = RLTrainer(**trainer_args)
    trainer.train()

if __name__ == '__main__':
    trainings = Parallel(n_jobs=-1)(delayed(train_run)(run) for run in range(50))
