import numpy as np
import random

class CaseNode:
    
    def __init__(self, n_obs, n_spec, n_color, cost_obs, cost_spec, cost_col, cost_coeff, all_features, all_clf):
        self.n_obs = n_obs
        self.n_spec = n_spec
        self.n_color = n_color
        self.settings = {'photo':n_obs, 'spectrum':n_spec, 'color':n_color}
        self.max_obs = all_features.lengths_gaia
        self.label = all_features.label
        self.cost_coeff = cost_coeff
        self.all_features = all_features
        self.cost_obs = cost_obs
        self.cost_spec = cost_spec
        self.cost_col = cost_col
        self.all_clf = all_clf
        self.survey_id = all_features.id_gaia
        
        obs_err = '"n_obs" must be lower than the number of available photometric \
                  observations for object {}: {}.'.format(self.survey_id, self.max_obs)

        assert (0 <= self.n_spec) & (self.n_spec <= 1), '"n_spec" must be {0,1}'
        assert (0 <= self.n_color) & (self.n_color <= 1), '"n_color" must be {0,1}'
        assert (0 < self.n_obs) & (self.n_obs <= self.max_obs), obs_err

        self.set_name()
        self.set_clf()
        self.set_features()
        self.set_x()
        self.set_prediction()
        self.set_cost()
        self.set_reward()
        
        end_cond = [self.max_obs == self.n_obs,
                    self.n_spec == 1,
                    self.n_color == 1]
        
        self.is_end = all(end_cond)
        self.best_child = None
        
    def set_reward(self):
        self.reward = self.label_prob - self.cost_coeff*self.cost
        
    def set_prediction(self):
        label_index = np.where(self.clf.classes_== self.label)[0][0]
        self.prediction = self.clf.predict([self.x])[0]
        self.label_prob = self.clf.predict_proba([self.x])[0][label_index]
        self.acc = int(self.label==self.prediction)

    def get_ts_features(self):
        prefix = 'h_'+str(self.n_obs-1) + '_'
        cols = list(filter(lambda c: prefix in c, self.features_names))
        features = self.all_features[cols].values
        return features

    def get_spec_features(self):
        features = None
        if self.n_spec==1:
            filter_ = lambda c: ('_sdss' in c) & (not 'id' in c)
            cols = list(filter(filter_, self.features_names))
            features = self.all_features[cols].values
        return features

    def get_c_features(self):
        features = None
        if self.n_color==1:
            features = self.all_features['color_gaia']
        return features

    def set_features(self):
        self.features_names = self.all_features.index.values
        self.ts_features = self.get_ts_features()
        self.spec_features = self.get_spec_features()
        self.c_features = self.get_c_features()

        
    def set_x(self):
        features_case = {'photo':self.ts_features, 'spec':self.spec_features, 'color':self.c_features}
        self.x = [features_case[key] for key in self.clf_features]
        self.x = np.hstack(self.x)
        
    def set_clf(self):
        cond0 = self.all_clf['spec']==self.n_spec
        cond1 = self.all_clf['color']==self.n_color
        cond2 = self.all_clf['photo']==bool(self.n_obs)
        filter_ = cond0 & cond1 & cond2
        assert sum(filter_) == 1

        self.clf_item = self.all_clf[filter_].iloc[0]
        # self.clf_features = self.clf_item[['photo', 'spec', 'color']] == 1
        self.clf_features = self.clf_item[self.clf_item==1].index.values
        self.clf = self.clf_item['clf']
        
    def set_name(self):
        self.name = 'ts'
        if self.n_obs == self.max_obs:
            self.name = self.name + '_full'
        else:
            self.name = self.name + '_' + str(self.n_obs)
        if self.n_color:
            self.name = self.name + '_' + 'col'
        if self.n_spec:
            self.name = self.name + '_' + 'spec'
    
    def set_cost(self):
        self.cost = self.n_obs*self.cost_obs + self.n_spec*self.cost_spec + self.n_color*self.cost_col

    def ctor_args(self):
        '''Get constructor arguments'''
        args = {'n_obs': self.n_obs,
                'n_spec': self.n_spec, 
                'n_color': self.n_color,
                'cost_obs': self.cost_obs, 
                'cost_spec': self.cost_spec, 
                'cost_col': self.cost_col, 
                'cost_coeff': self.cost_coeff,
                'all_features': self.all_features, 
                'all_clf': self.all_clf
               }
        return args

        
class BruteNode(CaseNode):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.children = []

class CasesTree:
    
    def __init__(self, cost_obs, cost_spec, cost_col, cost_coeff, all_features, all_clf, p_thres, strategy_name):
        self.all_features = all_features
        self.all_clf = all_clf
        self.survey_id = all_features.id_gaia
        
        self.cost_obs = cost_obs
        self.cost_spec = cost_spec
        self.cost_col = cost_col
        self.cost_coeff = cost_coeff
        self.label = all_features.label
        self.p_thres = p_thres
        self.strategy_name = strategy_name
        
        self.settings = {'cost_obs' : self.cost_obs,
                         'cost_spec' : self.cost_spec,
                         'cost_col' : self.cost_col,
                         'all_features': self.all_features, 
                         'all_clf': self.all_clf,
                         'cost_coeff': self.cost_coeff
                        }
        
    def create_root(self, n_obs, n_spec, n_color):
        case_args = self.settings.copy()
        case_args['n_obs'] = n_obs
        case_args['n_spec'] = n_spec
        case_args['n_color'] = n_color
        self.root = CaseNode(**case_args)
        
    def path_info(self):
        node = self.root
        self.costs = []
        self.probs = []
        self.rewards = []
        self.settings = []
        while not node is None:
            self.leaf = node
            self.costs.append(node.cost)
            self.probs.append(node.label_prob)
            self.settings.append(node.settings)
            self.rewards.append(node.reward)
            node = node.best_child


        self.steps_names = []
        for sett, sett_next in list(zip(self.settings[:-1], self.settings[1:])):
            added = [(k, sett[k]!=sett_next[k]) for k in sett.keys()]
            added = list(filter(lambda d: d[1], added))
            assert len(added)==1
            added_name = '+' + added[0][0]
            self.steps_names.append(added_name)
            
    def strategy_info(self):
        prob = self.probs[-1]
        reward = self.rewards[-1]
        cost = self.costs[-1]
        settings = self.settings[-1]
        last_node = self.root
        for i in range(len(self.rewards)-1):
            last_node = last_node.best_child
            
        prediction = last_node.prediction
        
        info = {'gaia_id':self.survey_id, 
                'strategy':self.strategy_name, 
                'probability':prob, 
                'reward':reward, 
                'cost':cost,
                'prediction': prediction,
                'label': self.label,
                'n_photo': settings['photo'],
                'n_spec': settings['spectrum'],
                'n_color': settings['color']
               }
        
        return info
        
    def plot_path(self, ax, label, color='r'):
        x = np.array(self.costs)
        y = np.array(self.probs)
        x0 = x[:-1]
        x1 = x[1:]
        y0 = y[:-1]
        y1 = y[1:]
        xpos = (x0+x1)/2
        ypos = (y0+y1)/2
        xdir = x1-x0
        ydir = y1-y0

        for X,Y,dX,dY in zip(xpos, ypos, xdir, ydir):
            ax.annotate("", xytext=(X,Y), xy=(X+0.001*dX,Y+0.001*dY), size=8,
                        arrowprops=dict(arrowstyle="->", color=color))#, alpha=0.5))
        
        y_delta = np.random.uniform(0.0, 0.02)
        for x_,y_,name in zip(x[1:], y[1:], self.steps_names):
            ax.annotate(name, (x_-0.15, y_+y_delta), size=8, color=color)

        ax.plot(x, y, linestyle='--', linewidth=0.9, color=color, zorder=1)#, alpha=0.5)
        ax.scatter(x, y, color=color, zorder=2, label=label)
        ax.set_xlabel('Cost')
        ax.set_ylabel('Ground class probability')
        
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xticks(np.arange(int(xmin), int(xmax)))
        ax.grid(linestyle='--')

        ax.legend()
        
    def label_line(line, label, x, y, color='0.5', size=12):
        xdata, ydata = line.get_data()
        x1 = xdata[0]
        x2 = xdata[-1]
        y1 = ydata[0]
        y2 = ydata[-1]

        ax = line.axes
        text = ax.annotate(label, xy=(x, y), xytext=(-10, 0),
                           textcoords='offset points',
                           size=size, color=color,
                           horizontalalignment='left',
                           verticalalignment='bottom',
                           zorder=-1)

        sp1 = ax.transData.transform_point((x1, y1))
        sp2 = ax.transData.transform_point((x2, y2))

        rise = (sp2[1] - sp1[1])
        run = (sp2[0] - sp1[0])

        slope_degrees = np.degrees(np.arctan2(rise, run))
        text.set_rotation(slope_degrees)
        return text
    
    def plot_reward_contour(self, ax, reward, color='lightskyblue'):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        prob_fn = lambda cost: reward + self.cost_coeff*cost
        x0 = xmin
        y0 = prob_fn(xmin)
        if y0 < ymin:
            y0 = ymin
            x0 = (y0-reward)/self.cost_coeff
        x1 = xmax
        y1 = prob_fn(xmax)
        if y1 > ymax:
            y1 = ymax
            x1 = (y1-reward)/self.cost_coeff

        x = [x0, x1]
        y = [y0, y1]
        x_middle = np.mean(x)
        y_middle = np.mean(y)
        line, = ax.plot(x, y, '--', color=color, linewidth=0.9)
        CasesTree.label_line(line, 'Reward = '+'{:.2f}'.format(reward), x_middle, y_middle, color=color)
        
class GreedyTree(CasesTree):
    
    def __init__(self, greedy_on, **kwargs):
        super().__init__(**kwargs)
        if greedy_on == 'probability':
            self.greedy_key = 'label_prob'
        elif greedy_on == 'reward':
            self.greedy_key = 'reward'
        else:
            raise AttributeError('CaseNode does not contain "greedy_on" attribute')
        self.best_value = None
        self.best_path = None
        self.curr_path = []
    
    def greedy_compare(self, case, best_candidate):
        if best_candidate is None:
            return case
        if getattr(case, self.greedy_key) > getattr(best_candidate, self.greedy_key):
            return case
        return best_candidate

    def greedy_path(self, case):
        self.curr_path = self.curr_path + [case]

        case_value = getattr(case, self.greedy_key)
        if (self.best_value is None) or (case_value > self.best_value):
            self.best_value = case_value
            self.best_path = self.curr_path.copy()

        if case.is_end:
            return
        
        args = self.settings.copy()
        args['n_obs'] = case.n_obs
        args['n_spec'] = case.n_spec
        args['n_color'] = case.n_color
        
        best_child = None
        
        if case.n_obs < case.max_obs:
            args_0 = args.copy() 
            args_0['n_obs'] = case.n_obs + 1
            child = CaseNode(**args_0)
            best_child = self.greedy_compare(child, best_child)
            
        if not case.n_spec:
            args_1 = args.copy() 
            args_1['n_spec'] = 1
            child = CaseNode(**args_1)
            best_child = self.greedy_compare(child, best_child)
        
        if not case.n_color:
            args_2 = args.copy() 
            args_2['n_color'] = 1
            child = CaseNode(**args_2)
            best_child = self.greedy_compare(child, best_child)
            
        self.greedy_path(best_child)
            
    def build_greedy(self):
        self.greedy_path(self.root)
        for parent, child in zip(self.best_path[:-1], self.best_path[1:]):
            parent.best_child = child
        self.path_info()
        
class PhotoTree(CasesTree):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def photo_path(self):
        args = self.settings.copy()       
        args['n_spec'] = 0
        args['n_color'] = 0
        
        case = self.root
        curr_path = [case]
        best_path = curr_path.copy()
        best_prob = case.label_prob
        
        while case.n_obs < case.max_obs:
            args['n_obs'] = case.n_obs + 1
            child = CaseNode(**args)
            curr_path = curr_path + [child]
            if child.label_prob > best_prob:
                best_prob = child.label_prob
                best_path = curr_path.copy()
            case = child
        
        for parent, child in zip(best_path[:-1], best_path[1:]):
            parent.best_child = child   
    
    def build_tree(self):
        self.photo_path()
        self.path_info()
        
class BruteForceTree(CasesTree):
    # Exhaustive strategy
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_prob = -1
        self.best_cost = np.inf
        self.best_path = None
        self.best_reward = -np.inf
        self.curr_path = []
        
    def create_root(self, n_obs, n_spec, n_color):
        case_args = self.settings.copy()
        case_args['n_obs'] = n_obs
        case_args['n_spec'] = n_spec
        case_args['n_color'] = n_color
        self.root = BruteNode(**case_args)
    
    def make_children(self, case):
        args = self.settings.copy()
        args['n_obs'] = case.n_obs
        args['n_spec'] = 0
        args['n_color'] = 0
         
        child0_args = args.copy()
        child0_args['n_spec'] = 1
        child0 = BruteNode(**child0_args)
        case.children.append(child0)
        
        child1_args = args.copy()
        child1_args['n_color'] = 1
        child1 = BruteNode(**child1_args)
        case.children.append(child1)
        
        grand_args = args.copy()
        grand_args['n_spec'] = 1
        grand_args['n_color'] = 1
        grand = BruteNode(**grand_args)
        child0.children.append(grand)
        child1.children.append(grand)
        
    def make_paths(self):
        case = self.root
        self.make_children(case)
        while case.n_obs < case.max_obs:
            args = self.settings.copy()
            args['n_obs'] = case.n_obs + 1
            args['n_spec'] = 0
            args['n_color'] = 0
            child = BruteNode(**args)
            case.children.append(child)
            case = child
            self.make_children(case)
            
    def dfs_rec(self, case):
        self.curr_path = self.curr_path + [case]
        # if case.label_prob > self.p_thres:
        #     if case.cost < self.best_cost:
        #         self.best_cost = case.cost
        #         self.best_prob = case.label_prob
        #         self.best_path = self.curr_path
        #     if (case.cost == self.best_cost) and self.best_prob < case.label_prob:
        #         self.best_cost = case.cost
        #         self.best_prob = case.label_prob
        #         self.best_path = self.curr_path
        # if case.label_prob > self.p_thres:
        if case.reward > self.best_reward:
            self.best_path = self.curr_path.copy()
            self.best_reward = case.reward
        for child in case.children:
            self.dfs_rec(child)
        self.curr_path = self.curr_path[:-1]
    
    def dfs(self):
        self.curr_path = []
        self.dfs_rec(self.root)
        for parent, child in zip(self.best_path[:-1], self.best_path[1:]):
            parent.best_child = child
        
    def build_tree(self):
        self.make_paths()
        self.dfs()
        self.path_info()
        
    def plot_path(self, ax, label, color='r'):
        super().plot_path(ax, label, color)
        ax.axhline(self.p_thres, color=color, linewidth=0.8)
        xmin, xmax = ax.get_xlim()
        x = (xmin + xmax)/2
        y = self.p_thres
        ax.text(x, y+0.01, 'Probability threshold = '+str(self.p_thres), color=color, size=12)

class RandomTree(BruteForceTree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def dfs_rec(self, case):
        self.curr_path = self.curr_path + [case]
        if case.label_prob > self.p_thres:
            self.candidate_paths.append(self.curr_path.copy())
        for child in case.children:
            self.dfs_rec(child)
        self.curr_path = self.curr_path[:-1]
    
    def dfs(self):
        self.curr_path = []
        self.candidate_paths = []
        self.dfs_rec(self.root)
        self.best_path = random.choice(self.candidate_paths)
        for parent, child in zip(self.best_path[:-1], self.best_path[1:]):
            parent.best_child = child
        
    def build_tree(self):
        self.make_paths()
        self.dfs()
        self.path_info()

class RLTree(CasesTree):

    def __init__(self, rl_model, processor, **kwargs):
        super().__init__(**kwargs)
        self.rl_model = rl_model
        self.processor = processor
    
    def rl_path(self):
        """
        next_action : {0, 1, 2, 3}
            0 : query for one more photometric observation
            1 : query for spectroscopy
            2 : query for color
            3 : do not query more obervations
        """
        args = self.settings.copy()
        args['n_obs'] = self.root.n_obs
        args['n_spec'] = 0
        args['n_color'] = 0
        
        case = self.root
        curr_path = [case]
        done = False

        while not (done or case.is_end):
            case_input = self.processor.process_observation(case)
            case_pred = self.rl_model.predict(np.expand_dims(case_input, 0))
            next_action = np.argmax(case_pred, axis=-1)[0]
            print(next_action)
            if next_action == 0:
                args['n_obs'] = case.n_obs + 1
            elif next_action == 1:
                args['n_spec'] = 1
            elif next_action == 2:
                args['n_color'] = 1
            else:
                done = True
            child = CaseNode(**args)
            curr_path = curr_path + [child]
            case = child
        
        for parent, child in zip(best_path[:-1], best_path[1:]):
            parent.best_child = child   
    
    def build_tree(self):
        self.rl_path()
        self.path_info()
    