from glob import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def color_features(bp_dir, rp_dir):
    files = glob(bp_dir+'*.dat')
    ids = []
    colors = []

    for path in files:
        id_ = int(path.split('/')[-1].rstrip('.dat'))
        ts_bp = pd.read_csv(path)
        ts_bp = ts_bp.sort_values('time', axis=0).mag

        path = rp_dir + str(id_) + '.dat'
        id_ = int(path.split('/')[-1].rstrip('.dat'))
        ts_rp = pd.read_csv(path)
        ts_rp = ts_rp.sort_values('time', axis=0).mag

        color = np.mean(ts_bp - ts_rp)
        ids.append(id_)
        colors.append(color)

    scaler = MinMaxScaler()
    c_features = pd.DataFrame()
    c_features['color_gaia'] = colors
    c_features[c_features.columns] = scaler.fit_transform(c_features)
    c_features['id_gaia'] = ids
    c_features = c_features.set_index('id_gaia')
    
    return c_features
    
if __name__ == '__main__':
    ftrs_dir = '../../../../data/features'
    obs_dir = '../../../../data/observations'
    save_path = f'{ftrs_dir}/gaia-sdss/features_color.csv'
    
    bp_dir = f'{obs_dir}/gaia/xmatch/bands/BP/'
    rp_dir = f'{obs_dir}/gaia/xmatch/bands/RP/'
    
    color_features_ = color_features(bp_dir, rp_dir)
    color_features_.to_csv(save_path)