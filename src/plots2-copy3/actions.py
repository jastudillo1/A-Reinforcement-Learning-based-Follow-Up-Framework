import math
import numpy as np
import random
import torch

class Decider:

    def __init__(self, target_net, policy_net, bl_recommender, processor, n_actions):
        self.steps_done = 0
        self.target_action = self.best_action(target_net)
        self.policy_action = self.best_action(policy_net)
        self.bl_recommender = bl_recommender
        self.bl_action = self.bl_recommender.recommend_step
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions

        self.EPS_START = 0.99
        self.EPS_END = 0.01
        self.EPS_DECAY = 5000

    def legal_action(self, case, action):
        if (action==0) and (case.n_obs>=case.max_obs):
            return False 
        if (action==1) and (case.n_spec==1):
            return False
        if (action==2) and (case.n_color==1):
            return False
        return True

    def random_action(self, case):
        random_actions = np.random.choice([0,1,2,3], size=4, replace=False, p=[0.4,0.2,0.2,0.2])#p=[0.7,0.1,0.1,0.1])#
        for action in random_actions:
            if self.legal_action(case, action):
                return torch.tensor([[action]], device=self.device, dtype=torch.long)
        raise ValueError('No legal action for case with id:{}, case.is_end:{}'.format(case.survey_id, case.is_end))

    def best_action(self, model):
        def best_action_fn(case):
            state = self.processor.process_obs(case)
            with torch.no_grad():
                q_action = model(state)
                _, top_actions = q_action.topk(self.n_actions,-1)
                top_actions = top_actions[0]
                for action in top_actions:
                    if self.legal_action(case, action):
                        return action.view(1, 1)
                raise ValueError('No legal action for case with id:{}, case.is_end:{}'.format(case.survey_id, case.is_end))
        return best_action_fn

    def train_action(self, case):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)

        self.steps_done += 1
        if sample <= eps_threshold:
            with torch.no_grad():
                return self.random_action(case)
        else:
            with torch.no_grad():
                return self.policy_action(case)

    def update_bl(self, case, next_case, action, update_knn):
        self.bl_recommender.push_experience(case, next_case, action, update_knn=update_knn)

    def strategy_action(self, strategy):
        if strategy=='random':
            return self.random_action
        elif strategy=='baseline':
            return self.bl_action
        elif strategy=='rl':
            return self.target_action
