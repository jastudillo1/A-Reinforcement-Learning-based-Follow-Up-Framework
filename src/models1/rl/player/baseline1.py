import sys
sys.path.insert(1, '../../Data/scripts/strategies')
from itertools import product
import numpy as np
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from trees import CaseNode

class Recommender:
    
    def __init__(self, n_neighbors=1):
        '''
        Parameters
        ----------
        cases_groups: dict
            Dictionary with neighbour candidates cases grouped by the number of 
            photometric, spectrum and color observations (n_obs, n_spec, n_color).
            This would be the exploration space to decide which step is the best
            in a given scenario. Each group is a dictionary indexed by the survey_id
            of the corresponding astronomical object.
        '''

        self.memory = {}
        self.knn_models = {}
        self.n_neighbors = n_neighbors
    
    def push_experience(self, case_t, case_t1, action, update_knn=False):
        '''
        Parameters
        ----------
        id_: int
            Survey id of the astronomical object you are exploring.
        key_t1:
            Current observational settings (n_obs, n_spec, n_color).
        key_t1:
            Observational settings (n_obs, n_spec, n_color) to explore.
        '''
        sett_key = (case_t.settings['photo'], 
                    case_t.settings['spectrum'], 
                    case_t.settings['color'])

        delta = case_t1.reward - case_t.reward
        experience = [case_t.x, delta]

        if not sett_key in self.memory:
            self.memory[sett_key] = {}
        if not action in self.memory[sett_key]:
            self.memory[sett_key][action] = []
        self.memory[sett_key][action].append(experience)
        if update_knn:
            self.update_knn_method(sett_key, action)
                
    def update_knn_method(self, sett_key, action):
        '''Set KNN models for a given observational scenario (n_obs, n_spec, n_color)'''
        
        experiences = self.memory[sett_key][action]
        X = [s[0] for s in experiences]
        X = np.vstack(X)
        model = NearestNeighbors(n_neighbors=self.n_neighbors)
        model.fit(X)

        if not sett_key in self.knn_models:
            self.knn_models[sett_key] = {}

        self.knn_models[sett_key][action] = model
    
    def nn_experience(self, case, action):

        '''Nearest neighbour to case'''
        sett_key = (case.settings['photo'], 
                    case.settings['spectrum'], 
                    case.settings['color'])
        x = case.x
        if not sett_key in self.knn_models:
            err_msg = 'There is no experience in memory similar to \
                   case with id:{} and settings:{}'.format(case.survey_id, sett_key)
            # raise ValueError(err_msg)
            print(err_msg)
            return None, None
        if not action in self.knn_models[sett_key]:
            return None, None

        model = self.knn_models[sett_key][action]
        distances, indices = model.kneighbors(x.reshape(1,-1))
        experience = self.memory[sett_key][indices[0]][0]
        
        return experience
    
    def recommend_step(self, case):
        '''Recommend next step given current case'''

        options = [3]

        if case.n_obs<case.max_obs:
            options.append(0)
        if case.n_spec==0:
            options.append(1)
        if case.n_color==0:
            options.append(2)

        if len(options)==0:
            return 3 # do not query more

        actions_no_exp = []
        actions_w_exp = []
        rewards_w_exp = []

        for action in options:
            _, delta_reward = self.nn_experience(case, action)
            if delta_reward is None:
                actions_no_exp.append(action)
            elif delta_reward>0:
                actions_w_exp.append(action)
                rewards_w_exp.append(delta_reward)

        if (len(actions_w_exp)==0):
            if (len(actions_no_exp)==0): # if no experience
                return 3 # do not query more
            else:
                return np.random.choice(actions_no_exp) #choose randomly an action
        else:
            order = np.argsort(rewards_w_exp)
            top_actions = actions_w_exp[order]
            return top_actions[0]
        