import numpy as np
import pandas as pd

def append_sdss_id(xmatch):
    plate = [str(p).zfill(4) for p in xmatch.PLATE_sdss]
    fiber = [str(f).zfill(4) for f in xmatch.FIBERID_sdss]
    mjd = xmatch.MJD_sdss
    ids = ['spec-{}-{}-{}.fits'.format(p, m, f) for p,m,f in zip(plate, mjd, fiber)]
    xmatch['id_sdss'] = ids
    return xmatch

if __name__=='__main__':
    ctlg_dir = '../../../data/catalogues'
    ftrs_dir = '../../../data/features'
    spec_path = f'{ftrs_dir}/gaia-sdss/features_spectra.csv'
    rnn_path = f'{ftrs_dir}/gaia-sdss/features_rnn.csv'
    color_path = f'{ftrs_dir}/gaia-sdss/features_color.csv'
    xmatch_path = f'{ctlg_dir}/gaia-sdss/cross-match-labels.csv'
    save_path = f'{ftrs_dir}/gaia-sdss/features.csv'
    
    ftrs_spectra = pd.read_csv(spec_path)
    ftrs_rnn = pd.read_csv(rnn_path)
    ftrs_color = pd.read_csv(color_path)
    xmatch = pd.read_csv(xmatch_path)
    
    xmatch = append_sdss_id(xmatch)
    xmatch = xmatch[['source_id_gaia', 'id_sdss', 'best_class_name_gaia']]
    xmatch = xmatch.rename(columns={'best_class_name_gaia': 'label', 'source_id_gaia': 'id_gaia'})
    xmatch = xmatch.merge(ftrs_spectra, on='id_sdss', how='inner')
    xmatch = xmatch.merge(ftrs_rnn, on='id_gaia', how='inner')
    xmatch = xmatch.merge(ftrs_color, on='id_gaia', how='inner')

    features_cols = set(xmatch.columns) - set(['id_gaia', 'id_sdss', 'label'])
    reorder = ['id_gaia', 'id_sdss', 'label'] + sorted(list(features_cols))
    xmatch = xmatch[reorder]
    xmatch.to_csv(save_path, index=False)