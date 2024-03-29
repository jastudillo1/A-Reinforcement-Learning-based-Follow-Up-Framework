import glob
import random
import numpy as np
import os
from collections import namedtuple
from itertools import count
import json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from baseline1 import Recommender
from environment1 import Environment
from classifiers import include_splits, build_sets
from processor import Processor
from trees import CaseNode
from actions import Decider



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        # middle = inputs - (inputs-outputs)//2
        # self.fc1 = nn.Linear(inputs, middle)
        # self.bn1 = nn.BatchNorm1d(middle)
        self.fc1 = nn.Linear(inputs, 15)
        self.fc2 = nn.Linear(15, 12)
        self.fc3 = nn.Linear(12, 8)
        self.fc4 = nn.Linear(8, outputs)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class Models:

    def __init__(self, BATCH_SIZE, GAMMA, INPUTS, N_ACTIONS, MEM_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.INPUTS = INPUTS
        self.N_ACTIONS = N_ACTIONS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_models()
        self.memory = ReplayMemory(MEM_SIZE)

    def set_models(self):
        self.policy_net = DQN(self.INPUTS, self.N_ACTIONS).to(self.device)
        self.target_net = DQN(self.INPUTS, self.N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters())

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_target(self, path):
        torch.save(self.target_net.state_dict(), path)
        

class SetUp:

    def __init__(self, features_path, clf_path, splits_path, settings_path, reward_path):
        self.features_path = features_path
        self.clf_path = clf_path
        self.splits_path = splits_path
        self.settings_path = settings_path
        self.reward_path = reward_path
        self.set_up()

    def set_up(self):
        self.set_ftrs()
        self.set_processor()
        self.set_env()
        self.set_cases()
        self.set_target()

    def set_processor(self):
        input_order = ['photo', 'spec', 'color', 'n_obs', 'n_spec', 'n_color']
        train_dataset = build_sets(self.train_ftrs)
        self.processor = Processor(train_dataset, input_order)

    def set_ftrs(self):
        with open(self.splits_path, 'rb') as f:
            splits = pickle.load(f)
        features = pd.read_csv(self.features_path)
        train_ftrs = features.set_index('id_gaia').loc[splits['rl_train']].reset_index()
        train_ftrs, val_ftrs = train_test_split(train_ftrs, stratify=train_ftrs['label'],
                                                        test_size=0.2, random_state=0)
        self.train_ftrs = train_ftrs
        self.val_ftrs = val_ftrs
        self.test_ftrs = features.set_index('id_gaia').loc[splits['test']].reset_index()

    def set_target(self):
        target = pd.read_csv(self.reward_path)
        train_ids = self.train_ftrs.id_gaia
        val_ids = self.val_ftrs.id_gaia
        test_ids = self.test_ftrs.id_gaia
        self.train_target = target.set_index('gaia_id').loc[train_ids].reset_index().reward
        self.val_target = target.set_index('gaia_id').loc[val_ids].reset_index().reward
        self.test_target = target.set_index('gaia_id').loc[test_ids].reset_index().reward

    def set_env(self):
        clf_df = pd.read_pickle(self.clf_path)
        with open(self.settings_path) as f:
            settings = json.load(f)
        tree_args = settings['tree_args']
        env_args = [tree_args['cost_obs'], tree_args['cost_spec'], tree_args['cost_col'], tree_args['cost_coeff']]
        self.env = Environment(clf_df, *env_args)

    def to_cases(self, features_, env):
        cases_ = []
        for i, case_features in features_.iterrows():
            case_args = env.case_settings.copy()
            case_args['all_features'] = case_features
            case_i = CaseNode(**case_args)
            cases_.append(case_i)
        return cases_

    def set_cases(self):
        self.train_cases = self.to_cases(self.train_ftrs, self.env)
        self.val_cases = self.to_cases(self.val_ftrs, self.env)
        self.test_cases = self.to_cases(self.test_ftrs, self.env)
        
        
    def rl_kwargs(self):
        args = {
            'env':self.env, 
            'processor':self.processor, 
            
            'train_ftrs':self.train_ftrs, 
            'train_cases': self.train_cases,
            'train_target': self.train_target,
            
            'val_ftrs':self.val_ftrs, 
            'val_cases': self.val_cases,
            'val_target': self.val_target,
            
            'test_ftrs':self.test_ftrs, 
            'test_cases': self.test_cases,
            'test_target': self.test_target,
            }
        return args
        
class RLFramework:
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.processor = kwargs['processor']
        self.set_models()
        self.set_data(**kwargs)
        n_actions = self.env.action_space.n
        
        self.target_net = self.models.target_net
        self.policy_net = self.models.policy_net
        self.bl_recommender = Recommender(100)
        self.decider = Decider(self.target_net, self.policy_net, self.bl_recommender, self.processor, n_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_data(self, **kwargs):
        # Set train_ftrs, test_ftrs...
        for split_k in ['train', 'val', 'test']:
            for attr_k in ['ftrs', 'cases', 'target']:
                attr_name = f'{split_k}_{attr_k}'
                attr_val = kwargs[attr_name]
                setattr(self, attr_name , attr_val)
                
    def set_models(self):
        models_args = {
            'BATCH_SIZE': 256,
            'GAMMA': 1,
            'INPUTS': 18,
            'N_ACTIONS': self.env.action_space.n,
            'MEM_SIZE':50000
            }
        
        self.models = Models(**models_args)
        
    def format_reward(self, reward):
        return torch.tensor(np.array([reward]).astype(np.float32), device=self.device)
        
    def take_action(self, case, state, action, stage):
        next_case, reward, done = self.env.step(action.item())
        reward = self.format_reward(reward)

        if not done:
            next_state = self.processor.process_obs(next_case)
        else:
            next_state = None

        # Store the transition in memory
        self.models.memory.push(state, action, next_state, reward)

        if stage == 'fill':
            self.decider.update_bl(case, next_case, action, update_knn = False)
        elif stage == 'train':
            self.decider.update_bl(case, next_case, action, update_knn = True)

        return next_case, next_state, done
        
    def restart(self, features):
        case = self.env.reset(features)
        state = self.processor.process_obs(case)

        return case, state
        
    def get_rewards(self, set_key, strategy_key, full_path=False):
        strategy_key = strategy_key.lower()
        assert strategy_key in ['random', 'baseline', 'rl', 'photo', 'photo_color', 
            'photo_spec', 'full']
        
        if set_key == 'train':
            dataset = self.train_cases
        elif set_key == 'val':
            dataset = self.val_cases
        elif set_key == 'test':
            dataset = self.test_cases
        else:
            raise KeyError('Expected either `train` or `val` for `set_key` argument.')
        
        action_fn = self.decider.strategy_action(strategy_key)
        rewards_ = []
        cases_end = []
        paths_end = []
        for case in dataset:
            self.env.case_t = case
            if full_path:
                case_path = [case]
            curr_case = case
            curr_state = self.processor.process_obs(curr_case)
            done = False
            while not done:
                action = action_fn(curr_case)
                next_case, reward, done = self.env.step(action.item())
                next_state = self.processor.process_obs(next_case)
                curr_case, curr_state = next_case, next_state
                if full_path:
                    case_path.append(curr_case)
            cases_end.append(curr_case)
            rewards_.append(curr_case.reward)    
            if full_path:
                paths_end.append(case_path)
        if full_path:
            return cases_end, rewards_, paths_end
        return cases_end, rewards_
        
class RLEval(RLFramework):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_model(kwargs['model_path'])
        self.FILL_MEM = 1000#20000
        self.fill_update_memory()
        
    def load_model(self, model_path):
        self.target_net.load_state_dict(torch.load(model_path))

    def restart(self):
        rand_index = np.random.choice(self.train_ftrs.shape[0])
        case_features = self.train_ftrs.iloc[rand_index]
        return super().restart(case_features)
        
    def fill_update_memory(self):
        
        for i in tqdm(range(self.FILL_MEM)):
            case, state = self.restart()
            for t in count():
                action = self.decider.random_action(case, p=[0.5,0.2,0.2,0.1])
                case, state, done = self.take_action(case, state, action, 'train')
                if done:
                    break
    
        
class RLTrainer(RLFramework):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run = kwargs['run']
        self.set_paths()
        
        self.TARGET_UPDATE = 50
        self.VAL_METRICS = 100
        self.FILL_MEM = 3000
        self.NUM_EPISODES = 6000

        self.ep_rewards = []
        self.rewards = {
            'rl':{'train':[], 'val':[]}, 
            'random':{'train':[], 'val':[]}, 
            'baseline':{'train':[], 'val':[]}, 
            'episode':[],
            'target': {'train':self.train_target.mean(), 'val':self.val_target.mean()}
            }
        
    def set_paths(self):
        self.model_dir = f'./runs/models'
        self.rewards_dir = f'./runs/rewards'
        self.summary_path = f'{self.model_dir}/summary.pk'
        self.rewards_path = f'{self.rewards_dir}/rewards_{self.run}.pk'
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.rewards_dir, exist_ok=True)
            
        self.model_name = f'model_{self.run}'
        self.model_path = f'{self.model_dir}/{self.model_name}.bin'


    def restart(self):
        rand_index = np.random.choice(self.train_ftrs.shape[0])
        case_features = self.train_ftrs.iloc[rand_index]
        return super().restart(case_features)
        
    def del_oldest(self):
        models_f = glob.glob(f'{self.model_dir}/model_*.bin')
        if len(models_f)>3:
            oldest_f = min(models_f, key=os.path.getctime)
            os.remove(oldest_f)
            
    def load_summ(self):
        if not os.path.exists(self.summary_path):
            summary = {}
        else:
            with open(self.summary_path, 'rb') as f:
                summary = pickle.load(f)
        return summary
    
    def update_summ(self, summary, episode):
        summary[self.model_name] = {
            'rewards':{
                'random': self.rewards['random']['val'][-1], 
                'baseline': self.rewards['baseline']['val'][-1], 
                'rl': self.rewards['rl']['val'][-1]},
            'episode': episode,
            'target': {'train':self.train_target.mean(), 'val':self.val_target.mean()}
            }
                                
        with open(self.summary_path, 'wb') as f:
            pickle.dump(summary, f)
        

    def check_save(self, episode):
        summary = self.load_summ()
        
        if len(summary)>0:
            best_rwd = max([model['rewards']['rl'] for model in summary.values()])
        else:
            best_rwd = -np.inf
        
        curr_rwd = self.rewards['rl']['val'][-1]
        if curr_rwd > best_rwd:
            self.models.save_target(self.model_path)
            self.del_oldest()
            self.update_summ(summary, episode)
                
        with open(self.rewards_path, 'wb') as f:
            pickle.dump(self.rewards, f)


    def val_metrics(self, episode):

        for strategy_key in ['random', 'baseline', 'rl']:
            _, rewards_train = self.get_rewards('train', strategy_key)
            _, rewards_val = self.get_rewards('val', strategy_key)
            self.rewards[strategy_key]['train'].append(np.mean(rewards_train))
            self.rewards[strategy_key]['val'].append(np.mean(rewards_val))
        step = self.decider.steps_done
        self.rewards['episode'].append(episode)
        self.check_save(episode)

    def fill_memory(self):
        j = 0
        for i in tqdm(range(self.FILL_MEM)):
            case, state = self.restart()
            for t in count():
                action = self.decider.random_action(case)
                case, state, done = self.take_action(case, state, action, 'fill')
                if done:
                    break
    
    def train(self):
        self.fill_memory()
        
        for episode in range(self.NUM_EPISODES):
            case, state = self.restart()
            for t in count():
                action = self.decider.train_action(case)
                case, state, done = self.take_action(case, state, action, 'train')
                self.models.optimize()
                if done:
                    self.ep_rewards.append(case.reward)
                    break

            if episode % self.TARGET_UPDATE == 0:
                self.models.update_target()

            if episode % self.VAL_METRICS == 0:
                self.val_metrics(episode)


        

# if __name__=='__main__':
    # data_dir = './data/'
    # features_path = data_dir + 'features-2.csv'
    # clf_path =  data_dir + 'classifiers.pkl'
    # rl_ids_path = data_dir + 'rl_ids.csv'
    # test_ids_path = data_dir + 'test_ids.csv'
    # settings_path = data_dir + 'settings8.json'
    # reward_path = data_dir + 'exhaustive_info.csv'

    # setup = SetUp(features_path, clf_path, rl_ids_path, test_ids_path, settings_path, reward_path)
    # trainer_args = setup.trainer_args()
    # trainer_args['run'] = 1
    # trainer = RLTrainer(**trainer_args)
    # trainer.train()

    # print('Complete')
