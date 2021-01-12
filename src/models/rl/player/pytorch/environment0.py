import numpy as np
from trees import CaseNode

class ActionSpace:

    def __init__(self):
        self.keys = range(4)
        self.n = len(self.keys)

    def sample(self):
        np.random.choice(self.keys)


class Environment:
    """
    Object for simulating observation query path
    
    case_t : `CaseNode`
        Holds the representation of the current state (features) and the 
        accumulated reward so far (not discounted).
    """
    
    def __init__(self, features, all_clf, cost_obs=1, cost_spec=5, cost_col=1, cost_coeff=0.025):
        """
        Parameters
        ----------
        all_clf : pandas.DataFrame
            Contains all classifiers for any given state (any number of photometric, 
            spectroscopic or color observations)
        """
        
        self.action_space = ActionSpace()
        self.min_obs = 5
        self.all_clf = all_clf
        self.case_t = None
        self.cost_obs = cost_obs
        self.cost_spec = cost_spec
        self.cost_col = cost_col
        self.cost_coeff = cost_coeff
        self.features = features
        self.case_settings = {
            'n_obs': self.min_obs, 
            'n_spec': 0, 
            'n_color': 0,
            'cost_obs': self.cost_obs, 
            'cost_spec': self.cost_spec, 
            'cost_col': self.cost_col, 
            'cost_coeff': self.cost_coeff,
            'all_clf': self.all_clf
            }

    def step(self, action):
        """
        Execute one of the actions to further observe an astronomical object.

        Parameters
        ----------
        action : {0, 1, 2, 3}
            0 : query for one more photometric observation
            1 : query for spectroscopy
            2 : query for color
            3 : do not query more obervations

        Returns
        -------
        case_t1 : `CaseNode`
            Resulting `CaseNode` after executing `action`
        reward : 
            The difference in the accumulated reward between the current state
            and the state after executing the action
        terminal :
            Either no more queries are required (action=3) or it is not possible 
            to execute this action anymore (no more observational resources 
            available)
        """
        
        if not action in self.action_space.keys:
            raise KeyError('`action` {} not within possible actions: {}'.format(action, self.actions))

        n_obs = self.case_t.n_obs
        n_spec = self.case_t.n_spec
        n_color = self.case_t.n_color

        case_t1 = self.case_t
        reward = 0
        done = False
        change = False

        if self.case_t.is_end or (action==3):
            done = True
        elif (action == 0) & (n_obs<self.case_t.max_obs):
            n_obs = n_obs + 1
            change = True
        elif (action == 1) & (n_spec == 0):
            n_spec = 1
            change = True
        elif (action == 2) & (n_color == 0):
            n_color = 1
            change = True

        if change:
            new_args = self.case_t.ctor_args()
            new_args.update({'n_obs': n_obs, 'n_spec': n_spec, 'n_color': n_color})
            case_t1 = CaseNode(**new_args)
            reward = case_t1.reward - self.case_t.reward
            done = case_t1.is_end#(case_t1.n_spec) and (case_t1.n_color)

        self.case_t = case_t1

        return case_t1, reward, done

    def reset(self):

        index = np.random.choice(self.features.shape[0])
        case_features = self.features.iloc[index]
        case_args = self.case_settings.copy()
        case_args['all_features'] = case_features
        self.case_t = CaseNode(**case_args)

        return self.case_t