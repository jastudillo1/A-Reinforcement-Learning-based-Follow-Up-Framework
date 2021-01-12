`ctlg.py`: builds catalogues of gaia sources to train and evaluate RNN model which will be used for feature extraction at a later pahse.

`bands.py`: split color bands from gaia observations for all sources in gaia-sdss cross-match. 

`preprocess_gaia.py`: serializes gaia data from training catalogues.

`preprocess_xgaia.py`: serializes gaia data from cross-match catalogues. 

`preprocess.sh`: bash file for making serialization work in a cluster (`preprocess_gaia.py`)

`features.py`: extract features from Becker RNN model as well as from expert features from color bands.

