import os, numpy as np, requests

DATA_URL = "https://raw.githubusercontent.com/JeffersonLab/jlab_datascience_data/main/eICU_age.npy"
DATA_FNAME = "eICU_age.npy"

def maybe_get_data(data_dir='.'):
    fpath = os.path.join(data_dir, DATA_FNAME)
    if not os.path.exists(fpath):
        print("Downloading eICU_age.npy ...")
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        with open(fpath, 'wb') as f:
            f.write(r.content)
    ages = np.load(fpath)
    return ages.astype('float32')
