import numpy as np

def reshape(data):
    return np.reshape(data, ( data.shape[0], 28, 28, 1 ) )

