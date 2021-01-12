from astropy.io import fits
from glob import glob
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class SpectraDB:
    
    def get_features(self, path):
        with fits.open(path, memmap=False) as hdulist:
            K = hdulist[2].data.ELODIE_TEFF
            gravity = hdulist[2].data.ELODIE_LOGG
            feh = hdulist[2].data.ELODIE_FEH
            z = hdulist[2].data.ELODIE_Z

            fiber = hdulist[0].header['FIBERID']
            fiber = str(fiber).zfill(4)
            plate = hdulist[0].header['PLATEID']
            plate = str(plate).zfill(4)
            mjd = hdulist[0].header['MJD']
            id_ = 'spec-{}-{}-{}.fits'.format(plate, mjd, fiber)

        return K, gravity, feh, z, id_

    def build_df(self, spec_dir):
        paths = glob(f'{spec_dir}/*.fits')
        features = Parallel(n_jobs=-1)(delayed(self.get_features)(path) for path in tqdm(paths))
        K, gravity, feh, z, ids = list(zip(*features))
        scaler = MinMaxScaler()
        
        format_fn = lambda seq: np.array(seq).reshape(-1)
        df = pd.DataFrame()
        df['K'] = format_fn(K)
        df['gravity'] = format_fn(gravity)
        df['feh'] = format_fn(feh)
        df['z'] = format_fn(z)
        df['id'] = format_fn(ids)
        df = df.add_suffix('_sdss')
        df = df.set_index('id_sdss', drop=True)
        df[df.columns] = scaler.fit_transform(df)
        
        return df
        
if __name__ == '__main__':
    ftrs_dir = '../../../../data/features'
    obs_dir = '../../../../data/observations'
    spec_dir = f'{obs_dir}/sdss/xmatch'
    save_path = f'{ftrs_dir}/gaia-sdss/features_spectra.csv'
    
    spec_db = SpectraDB()
    spec_features = spec_db.build_df(spec_dir)
    spec_features.to_csv(save_path)