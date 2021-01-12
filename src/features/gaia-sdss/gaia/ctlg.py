import numpy as np
import os
import pandas as pd

def format_ctlg(gaia_ctlg, gaia_obs_dir):
    gaia_obs_dir = os.path.abspath(gaia_obs_dir)
    
    mkpath = lambda source_id: f'{gaia_obs_dir}/{source_id}.dat'
    gaia_ctlg = gaia_ctlg[['best_class_name', 'astrometric_matched_observations']]
    gaia_ctlg.rename(
        columns={'best_class_name':'Class', 'astrometric_matched_observations': 'N'},
        inplace=True)
    gaia_ctlg.index.name = 'ID'
    gaia_ctlg.reset_index(inplace=True)
    gaia_ctlg['Address'] = gaia_ctlg.ID.apply(mkpath)
    
    return gaia_ctlg

def gaia_rnn_ctlg(gaia_sources, xmatch, ckeep, min_N, gaia_obs_dir):
    '''
    Buils gaia catalogue for training RNN model. 
    
    The RNN model is meant to be used for gaia feature extraction (hidden states).
    (1) Filters out gaia observations in cross match to ensure independence of classification
    results in the final framework.
    (2) Filters sources with a minimum number of photometric observations.
    (3) Filters sources belonging to `ckeep` labels (meant to avoid training with small classes).
    
    The resulting catalogue includes absolute location to corresponding observations files 
    to be used in the serialization of the data that will be input into the RNN.
    
    Inputs
    ------
    gaia_sources: pd.DataFrame
        Catalogue of gaia sources. Must be indexed by ID and contain 
        `astrometric_matched_observations` and `best_class_name`.
    xmatch:  pd.DataFrame
        gaia-sdss cross-match catalogue.
    ckeep: list-like
        List iwth labels to keep.
    gaia_obs_dir: str
        Directory where gaia observations are located.
    '''
    
    in_xmatch = [idx in xmatch.source_id_gaia.values for idx in gaia_sources.index.values]
    out_xmatch = ~np.array(in_xmatch)
    in_N = gaia_sources.matched_observations >= min_N
    in_cls = gaia_sources.best_class_name.isin(ckeep)
    gaia_filter = gaia_sources[out_xmatch & in_N & in_cls]
    
    print('Original dataset size:', gaia_sources.shape[0])
    print('Shrinked dataset size:', gaia_filter.shape[0])
    
    gaia_filter = format_ctlg(gaia_filter, gaia_obs_dir)
    return gaia_filter
    
def xgaia_rnn_ctlg(gaia_sources, xmatch, ckeep, min_N, gaia_obs_dir):
    '''
    Buils gaia catalogue for gaia in cross-match feature extraction with RNN model. 
    '''
    
    in_xmatch = [idx in xmatch.source_id_gaia.values for idx in gaia_sources.index.values]
    in_N = gaia_sources.matched_observations >= min_N
    in_cls = gaia_sources.best_class_name.isin(ckeep)
    gaia_filter = gaia_sources[in_xmatch & in_N & in_cls]
    
    print('Original dataset size:', gaia_sources.shape[0])
    print('Shrinked dataset size:', gaia_filter.shape[0])
    
    gaia_filter = format_ctlg(gaia_filter, gaia_obs_dir)
    return gaia_filter
    
    
if __name__=='__main__':
    
    ctlg_dir = '../../../../data/catalogues'
    obs_dir = '../../../../data/observations'

    xmatch_path = f'{ctlg_dir}/gaia-sdss/cross-match-labels.csv'
    gaia_sources_path = f'{ctlg_dir}/gaia/gaia-variable-sources.csv'
    gaia_labels_path = f'{ctlg_dir}/gaia/gaia-vari-classifier-result.csv'
    gaia_obs_dir = f'{obs_dir}/gaia/G/'
    gaia_rnn_ctlg_path = f'{ctlg_dir}/gaia/gaia-rnn.dat'
    xgaia_rnn_ctlg_path = f'{ctlg_dir}/gaia-sdss/xgaia-rnn.dat'

    min_N = 15
    ckeep = ['RRC', 'RRD', 'DSCT_SXPHE', 'MIRA_SR', 'RRAB']
    
    xmatch = pd.read_csv(xmatch_path, usecols=['source_id_gaia'])

    gaia_sources = pd.read_csv(gaia_sources_path)
    gaia_sources = gaia_sources.set_index('source_id')

    gaia_labels = pd.read_csv(gaia_labels_path)
    gaia_labels = gaia_labels.set_index('source_id')

    gaia_sources = gaia_sources.join(gaia_labels['best_class_name'])
    
    # Gaia sources for RNN model training
    gaia_rnn_ctlg_ = gaia_rnn_ctlg(gaia_sources, xmatch, ckeep, min_N, gaia_obs_dir)
    gaia_rnn_ctlg_.to_csv(gaia_rnn_ctlg_path, index=False)
    
    # Gaia sources in cross-match for RNN feature extraction
    xgaia_rnn_ctlg_ = xgaia_rnn_ctlg(gaia_sources, xmatch, ckeep, min_N, gaia_obs_dir)
    xgaia_rnn_ctlg_.to_csv(xgaia_rnn_ctlg_path, index=False)