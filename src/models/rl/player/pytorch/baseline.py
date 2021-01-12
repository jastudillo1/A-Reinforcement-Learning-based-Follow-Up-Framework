import sys
sys.path.insert(1, '../../Data/scripts/strategies')
from itertools import product
import numpy as np
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from trees import CaseNode

class CasesArchive:

    def __init__(self, fset, setup_base):
        self.fset = fset
        self.setup_base = setup_base
        self.cases_groups = {} 

    def build_archive(self, obs_range, spec_range, col_range):
        self.cases_groups = {}

        n_sources = list(product(obs_range, spec_range, col_range))
        n_sources = sorted(n_sources, key=itemgetter(1))

        for n_sources_ in tqdm(n_sources):
            setup = CasesArchive.costumize_setup(*n_sources_, self.setup_base)
            self.cases_groups[n_sources_] = self.setup_cases(setup)

    def costumize_setup(n_obs, n_spec, n_color, setup_base):
        setup = setup_base.copy()
        setup['n_obs'] = n_obs
        setup['n_spec'] = n_spec
        setup['n_color'] = n_color

        return setup

    def setup_cases(self, setup):
        '''
        Simulates and builds the observational `cases` in `features` that meet the 
        specified `setup`.
        
        Parameteres
        -----------
        setup: dict
            Setup for building cases. 
        fset: pandas.DataFrame
            Features set. Each row represents an astronomical object and contains all 
            its featuresfor any given scenario (all possible n_obs, n_spec, n_color) 
            for that object.
            
        Returns
        -------
        cases: `CaseNode` list.
            A list of `CaseNode` with all the cases that meet the `setup` criteria.
        '''
        
        n_obs = setup['n_obs']
        cases = {}
        
        # Filter objects which have at least one further photometric observation.
        # Do not include objects which do not have possible further obersvational steps in the database.
        fset = self.fset[self.fset.lengths_gaia>=(n_obs+1)]
        for _, features in fset.iterrows():
            setup['all_features'] = features
            case = CaseNode(**setup)
            if case.survey_id in cases:
                raise ValueError('Duplicate entry for the same bject with gaia id:{}'.format(case.survey_id))
            if case.is_end:
                pass
            else:
                cases[case.survey_id] = case

        return cases

class Explorer:
    
    def __init__(self, cases_groups, n_neighbors=20):
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
        self.cases_groups = cases_groups
        self.memory = {}
        self.n_neighbors = n_neighbors
    
    def add_entry(self, id_, key_t, key_t1, step_name):
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
        if key_t1 in self.cases_groups: 
            case_t = self.cases_groups[key_t][id_]
            case_t1 = self.cases_groups[key_t1][id_]
            delta = case_t1.reward - case_t.reward
            experience = [case_t.x, step_name, delta]
            self.memory[key_t].append(experience)
    
    def explore(self, case):
        '''
        Adds to memory the experience of each possible further observational 
        step in the current case.
        
        Parameters
        ----------
        case: CaseNode
        '''
        
        id_ = case.survey_id
        key_t = (case.settings['photo'], 
                 case.settings['spectrum'], 
                 case.settings['color'])
        
        steps = {}
        steps['photo'] = (key_t[0] + 1, key_t[1], key_t[2])
        steps['spectrum'] = (key_t[0], key_t[1] + 1, key_t[2])
        steps['color'] = (key_t[0], key_t[1], key_t[2] + 1)
        
        for step_name, key_t1 in steps.items():
            if (key_t1 in self.cases_groups) and (id_ in self.cases_groups[key_t1]):
                self.add_entry(id_, key_t, key_t1, step_name)
    
    def fill_memory(self):
        '''
        Explore and set the best next observational step for each of the cases 
        in the available cases in the database (self.cases_groups).
        '''
        for sett, cases in self.cases_groups.items():
            self.memory[sett] = []
            counter = 0
            for case in cases.values():
                self.explore(case)
                counter +=1
            self.memory[sett] = np.array(self.memory[sett])
            if len(self.memory[sett])<self.n_neighbors:
                del self.memory[sett]
                
    def set_knn(self):
        '''Set KNN models for each observational scenario (n_obs, n_spec, n_color)'''
        self.knn_models = {}
        for key, steps in self.memory.items():
            X = steps[::,0]
            X = np.vstack(X)
            model = NearestNeighbors(n_neighbors=self.n_neighbors)
            model.fit(X)
            self.knn_models[key] = model
    
    def nn_experience(self, case):
        '''Nearest neighbour to case'''
        sett_key = (case.settings['photo'], 
                    case.settings['spectrum'], 
                    case.settings['color'])
        x = case.x
        if not sett_key in self.knn_models:
            err_msg = 'There is no experience in memory similar to \
                   case with id:{} and settings:{}'.format(case.survey_id, sett_key)
            raise ValueError(err_msg)
        model = self.knn_models[sett_key]
        distances, indices = model.kneighbors(x.reshape(1,-1))
        experience = self.memory[sett_key][indices[0]]
        
        return experience
    
    def recommend_step(self, case):
        '''Recommend next step given current case'''
        experience = self.nn_experience(case)
        step_type = experience[::,1]
        
        best_step = None
        best_reward = None
        options = ['photo']
        if case.n_spec==0:
            options.append('spectrum')
        if case.n_color==0:
            options.append('color')
        for type_ in options:
            filter_ = np.where(step_type==type_)[0]
            if len(filter_)==0:
                return None, None
            filter_reward = np.mean(experience[filter_][::,2])
            if (best_step is None) or (filter_reward>best_reward):
                best_step = type_
                best_reward = filter_reward
        return best_step, best_reward