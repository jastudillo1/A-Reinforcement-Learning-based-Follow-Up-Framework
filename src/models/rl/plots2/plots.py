import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

import sys
from baseline1 import Recommender
from environment1 import Environment
from classifiers import include_splits, build_sets
from processor import Processor
from trees import CaseNode


# load Data
data_dir = './data/'
features_path = data_dir + 'features-2.csv'
clf_path =  data_dir + 'classifiers.pkl'
rl_ids_path = data_dir + 'rl_ids.csv'
test_ids_path = data_dir + 'test_ids.csv'
settings_path = data_dir + 'settings8.json'
reward_path = data_dir + 'exhaustive_info.csv'

splits = {'rl':rl_ids_path}
features = pd.read_csv(features_path)
train_features = include_splits(features, splits, drop_rows=True)
train_features, val_features = train_test_split(train_features, stratify=train_features['label'],
                                                test_size=0.2, random_state=0)
train_dataset = build_sets(train_features)
clf_df = pd.read_pickle(clf_path)
target_reward = pd.read_csv(reward_path)

with open(settings_path) as f:
    settings = json.load(f)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

BATCH_SIZE = 256
GAMMA = 1#0.999
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 3000
TARGET_UPDATE = 20
VAL_METRICS = 100
FILL_MEM = 5000
INPUTS = 18
input_order = ['photo', 'spec', 'color', 'n_obs', 'n_spec', 'n_color']

processor = Processor(train_dataset, input_order)
tree_args = settings['tree_args']
env_args = [tree_args['cost_obs'], tree_args['cost_spec'], tree_args['cost_col'], tree_args['cost_coeff']]
env = Environment(clf_df, *env_args)
n_actions = env.action_space.n

policy_net = DQN(INPUTS, n_actions).to(device)
target_net = DQN(INPUTS, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(1000000)

steps_done = 0

def legal_action(case, action):
    # if (case.n_spec) or (case.n_color):
    #     if action==0:
    #         return False
    if (action==0) and (case.n_obs>=case.max_obs):
        return False 
    if (action==1) and (case.n_spec==1):
        return False
    if (action==2) and (case.n_color==1):
        return False
    return True

def random_action(case):
    random_actions = np.random.choice([0,1,2,3], size=4, replace=False, p=[0.25,0.25,0.25,0.25])#p=[0.7,0.1,0.1,0.1])#
    for action in random_actions:
        if legal_action(case, action):
            return torch.tensor([[action]], device=device, dtype=torch.long)
    raise ValueError('No legal action for case with id:{}, case.is_end:{}'.format(case.survey_id, case.is_end))

def best_action(case, model):
    state = processor.process_observation(case)
    with torch.no_grad():
        q_action = model(state)
        _, top_actions = q_action.topk(n_actions,-1)
        top_actions = top_actions[0]
        for action in top_actions:
            if legal_action(case, action):
                return action.view(1, 1)
        raise ValueError('No legal action for case with id:{}, case.is_end:{}'.format(case.survey_id, case.is_end))

def select_action(case, validation=False, random_=False):
    
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if validation:
        with torch.no_grad():
            return best_action(case, target_net)

    elif random_ or sample <= eps_threshold:
        steps_done += 1
        with torch.no_grad():
            return random_action(case)
    else:
        steps_done += 1
        with torch.no_grad():
            return best_action(case, policy_net)

episode_rewards = []
episode_grounds = [] 
episode_diffs = [] 

val_rewards = []
val_rewards_rand = []
val_rewards_baseline = []

val_accs = []
val_accs_rand = []
val_accs_baseline = []

def plot_durations():
    plt.figure(2, figsize=(5,7))
    plt.clf()
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    ax = plt.subplot(2,1,1)
    ax.set_title('Training Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    N = 100
    if len(rewards_t) >= N:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax.plot(means.numpy(), label='Mean reward over last {} episodes'.format(N))
        ax.legend()

    ax = plt.subplot(2,1,2)
    ax.set_title('Validation Rewards every {} episodes'.format(VAL_METRICS))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    if len(val_rewards)>0:
        val_label = ['{:.2f}'.format(a) for a in val_accs]
        val_rand_label = ['{:.2f}'.format(a) for a in val_accs_rand]
        val_baseline_label = ['{:.2f}'.format(a) for a in val_accs_baseline]
        x = np.array(range(len(val_rewards)))
        ax.plot(x, val_rewards)
        ax.scatter(x, val_rewards, label='Optimal strategy reward')
        ax.scatter(x, val_rewards_rand, label='Random strategy rewards')
        ax.scatter(x, val_rewards_baseline, label='Baseline strategy rewards')
        for i, txt in enumerate(val_label):
            ax.annotate(txt, (x[i], val_rewards[i]))
        for i, txt in enumerate(val_rand_label):
            ax.annotate(txt, (x[i], val_rewards_rand[i]))
        for i, txt in enumerate(val_baseline_label):
            ax.annotate(txt, (x[i], val_rewards_baseline[i]))

        ax.set_xticks(x)
        xlabels = [str(i) for i in x*VAL_METRICS]
        ax.set_xticklabels(xlabels) 
        ax.axhline(y=np.mean(val_reward_target), color='r', linestyle='-', label='Max achievable reward')
        # ax.axhline(y=baseline_val, color='g', linestyle='-', label='Baseline reward')
        ax.annotate('Max achievable acc: {:.2f}'.format(val_acc_target), (x[0], np.mean(val_reward_target)))
        ax.legend()

    plt.subplots_adjust(hspace=0.5)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = torch.nn.MSELoss()(state_action_values.view(-1), expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def to_cases(features_, env):
    cases_ = []
    for i, case_features in features_.iterrows():
        case_args = env.case_settings.copy()
        case_args['all_features'] = case_features
        case_i = CaseNode(**case_args)
        cases_.append(case_i)
    return cases_

def sources_metrics(cases_, title=None):
    total = len(cases_)
    print('Total shape:', total)
    
    group_total = sum([case_i.acc for case_i in cases_])
    acc = group_total/total
    print('Accuracy: {:.2f}'.format(acc), ', Right:', group_total, ', Wrong:', total-group_total)
    
    group_total = sum([case_i.n_spec for case_i in cases_])
    rate = group_total/total
    print('spectrum:', group_total, '{:.3f}'.format(rate))

    group_total = sum([case_i.n_color for case_i in cases_])
    rate = group_total/total
    print('color:', group_total, '{:.3f}'.format(rate))

    group_total = sum([(bool(case_i.n_spec) & bool(case_i.n_color))for case_i in cases_])
    rate = group_total/total
    print('spectrum and color:', group_total, '{:.3f}'.format(rate))
    
    photo = [case_i.n_obs for case_i in cases_]
    xticks = list(range(5,25))
    values, counts = np.unique(photo, return_counts=True)
    counts = counts/sum(counts)
    plt.figure()
    plt.bar(values, counts)
    plt.xticks(ticks=xticks)
    plt.xlabel('Nº of observations')
    if title:
        plt.title(title)
    plt.show()

recommender = Recommender(100)

def get_reward_baseline(cases_):
    rewards_ = []
    accs_ = []
    cases_end = []
    count = 0
    for case_i in cases_:
        env.case_t = case_i
        curr_case = case_i
        curr_state = processor.process_observation(curr_case)
        done = False
        while not done:
            action = recommender.recommend_step(curr_case)
            ############# print action.item()
            next_case, reward, done = env.step(action.item())
            next_state = processor.process_observation(next_case)
            curr_case = next_case
            curr_state = next_state
        rewards_.append(curr_case.reward)
        accs_.append(curr_case.acc)
        cases_end.append(curr_case)
    return cases_end, rewards_, accs_

def get_reward(cases_):
    rewards_ = []
    accs_ = []
    cases_end = []
    count = 0
    for case_i in cases_:
        env.case_t = case_i
        curr_case = case_i
        curr_state = processor.process_observation(curr_case)
        done = False
        while not done:
            action = select_action(curr_case, validation=True)
            ############# print action.item()
            next_case, reward, done = env.step(action.item())
            next_state = processor.process_observation(next_case)
            curr_case = next_case
            curr_state = next_state
        rewards_.append(curr_case.reward)
        accs_.append(curr_case.acc)
        cases_end.append(curr_case)
    return cases_end, rewards_, accs_

def get_reward_rand(cases_):
    rewards_ = []
    accs_ = []
    cases_end = []
    count = 0
    for case_i in cases_:
        env.case_t = case_i
        curr_case = case_i
        curr_state = processor.process_observation(curr_case)
        done = False
        while not done:
            action = random_action(curr_case)
            next_case, reward, done = env.step(action.item())
            next_state = processor.process_observation(next_case)
            curr_case = next_case
            curr_state = next_state
        cases_end.append(curr_case)
        accs_.append(curr_case.acc)
        rewards_.append(curr_case.reward)
    return cases_end, rewards_, accs_

train_target = target_reward.set_index('gaia_id').loc[train_features['id_gaia'].values]
val_target = target_reward.set_index('gaia_id').loc[val_features['id_gaia'].values]
val_reward_target = val_target.reward.values
val_acc_target = (val_target.label==val_target.prediction).astype(int).mean()
val_cases = to_cases(val_features, env)

# actions_dict = {0: 'photo', 1:'spec', 2:'color', 3:'stop'}

# for i, case_features in fill_features.iterrows():
for i in tqdm(range(FILL_MEM)):
    rand_index = np.random.choice(train_features.shape[0])
    case_features = train_features.iloc[rand_index]
    case = env.reset(case_features)
    state = processor.process_observation(case)
    for t in count():
        # Select and perform an action
        action = select_action(case, random_=True)
        next_case, reward, done = env.step(action.item())

        reward = torch.tensor(np.array([reward]).astype(np.float32), device=device)

        # Observe new state
        if not done:
            next_state = processor.process_observation(next_case)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        recommender.push_experience(case, next_case, action, update_knn=False)

        # Move to the next state
        case = next_case
        state = next_state

        if done:
            break

num_episodes = 5000
for i_episode in range(num_episodes):
    rand_index = np.random.choice(train_features.shape[0])
    case_features = train_features.iloc[rand_index]
    case = env.reset(case_features)
    state = processor.process_observation(case)
    actions = []
    for t in count():
        # Select and perform an action
        action = select_action(case)
        actions.append(action.data.tolist()[0][0])
        next_case, reward, done = env.step(action.item())
        reward = torch.tensor(np.array([reward]).astype(np.float32), device=device)

        # Observe new state
        if not done:
            next_state = processor.process_observation(next_case)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        recommender.push_experience(case, next_case, action, update_knn=True)

        # Move to the next state
        case = next_case
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_rewards.append(case.reward)
            episode_grounds.append(train_target.loc[case.survey_id].reward)
            episode_diffs.append(episode_grounds[-1]-episode_rewards[-1])
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if i_episode % VAL_METRICS == 0:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        print(eps_threshold, steps_done, memory.capacity, memory.position, val_accs)
        final_val_cases, val_rewards_all, val_accs_all = get_reward(val_cases)
        val_rewards.append(np.mean(val_rewards_all))
        val_accs.append(np.mean(val_accs_all))
        sources_metrics(final_val_cases)

        '''Random'''
        _, val_rewards_rand_all, val_accs_rand_all = get_reward_rand(val_cases)
        val_rewards_rand.append(np.mean(val_rewards_rand_all))
        val_accs_rand.append(np.mean(val_accs_rand_all))

        '''Baseline'''
        _, val_rewards_baseline_all, val_accs_baseline_all = get_reward_baseline(val_cases)
        val_rewards_baseline.append(np.mean(val_rewards_baseline_all))
        val_accs_baseline.append(np.mean(val_accs_baseline_all))

print('Complete')
plt.ioff()
plt.show()