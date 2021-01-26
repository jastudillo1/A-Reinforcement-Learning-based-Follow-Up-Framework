import numpy as np
import torch

class Processor:
    
    def __init__(self, datasets, input_order):
        self.input_order = np.array(input_order)
        self.features_slices(datasets)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def slice_(self, key):
        key_i = np.where(self.input_order==key)[0][0]
        key_size = self.sizes[key]
        skip_i = range(0,max(key_i,0))
        init = 0
        for i in skip_i:
            skip_key = self.input_order[i]
            init += self.sizes[skip_key]
        end = init + key_size
        return slice(init, end)

    def features_slices(self, datasets):
        photo_shape = datasets['photo'][0].shape[0]
        spec_shape = datasets['spec'][0].shape[0]
        color_shape = datasets['color'][0].shape[0]
        nphoto_shape = 1
        nspec_shape = 1
        ncolor_shape = 1
        total_shape = photo_shape + spec_shape + color_shape + nphoto_shape + nspec_shape + ncolor_shape
        self.input_shape = total_shape

        self.sizes = {'photo':photo_shape, 
                      'spec':spec_shape, 
                      'color':color_shape,
                      'n_obs':nphoto_shape, 
                      'n_spec':nspec_shape, 
                      'n_color':ncolor_shape
                      }
        
        self.slices = {'photo':self.slice_('photo'), 
                       'spec':self.slice_('spec'), 
                       'color':self.slice_('color'), 
                       'n_obs':self.slice_('n_obs'), 
                       'n_spec': self.slice_('n_spec'),
                       'n_color':self.slice_('n_color')
                       }

    def process_obs(self, case):
        input_ = np.zeros(self.input_shape)
        if not case.ts_features is None:
            input_[self.slices['photo']] = case.ts_features
            input_[self.slices['n_obs']] = case.n_obs
        if not case.spec_features is None:
            input_[self.slices['spec']] = case.spec_features
            input_[self.slices['n_spec']] = case.n_spec
        if not case.c_features is None:
            input_[self.slices['color']] = case.c_features
            input_[self.slices['n_color']] = case.n_color

        input_ = np.expand_dims(input_.astype(np.float32), axis=0)
        
        return torch.from_numpy(input_).to(self.device)

    # def process_reward(self, reward):
    #     return np.clip(reward, -1., 1.)