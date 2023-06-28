import sys
import os
from keras.models import load_model
from attention import attention
import pickle
import FUNCTIONS as fn
import importlib

importlib.reload(fn)

WORK_DIR = "./"
ind = [0,1,2,3,4,5,7]

if __name__ == "__main__":
    data = fn.load_dataset(sys.argv[1])
    X_data = data['feature_data'][:,:,ind]
    tmp = X_data.reshape(-1, X_data.shape[2])
    mean = tmp.mean(axis=0)
    std = tmp.std(axis=0)
    X_data -= mean
    X_data /= std
    folder = sys.argv[2]
    model_file = "models/model.reg_trained.class.hdf5"
    weights_file = WORK_DIR + folder + "model_weights.class.hdf5"
    model = load_model(model_file, custom_objects={'attention': attention},compile=False)
    model.load_weights(weights_file)
    y_pred = model.predict([X_data[:, :, :4], X_data[:, :, 4], X_data[:, :, 5], X_data[:, :, 6]])
    with open(WORK_DIR + folder + 'predicted_classes.pickle', 'wb') as handle:
        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)


