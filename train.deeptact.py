import os
from attention import attention
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
import h5py
import tensorflow as tf


if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


config = tf.compat.v1.ConfigProto(log_device_placement=True, device_count={'GPU': 8, 'CPU': 1})

session = tf.compat.v1.Session(config=config)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


print("finished loading modules...")
INPUT_LENGTH = 1000
EPOCH = 200
BATCH_SIZE = 64
GPUS = 4
WORK_DIR = "./"

def run_model(data, model, save_dir):
    weights_file = os.path.join(save_dir, "model_weights.hdf5")
    model_file = os.path.join(save_dir, "single_model.hdf5")
    model.save(model_file)

    parallel_model = model
    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ind = [0,1,2,3,4,5,7]
    X_train = data["train_data"][:,:,ind]
    Y_train = data["train_labels"]
    X_validation = data["val_data"][:,:,ind]
    Y_validation = data["val_labels"]
    X_test = data["test_data"][:,:,ind]
    Y_test = data["test_labels"]

    # # normalize features to mean 0 and std 1
    tmp = X_train.reshape(-1, X_train.shape[2])
    mean = tmp.mean(axis=0)
    std = tmp.std(axis=0)
    X_test -= mean
    X_test /= std
    X_train -= mean
    X_train /= std
    X_validation -= mean
    X_validation /= std

    _callbacks = []
    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    _callbacks.append(checkpointer)
    earlystopper = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    _callbacks.append(earlystopper)

    parallel_model.fit([X_train[:,:,:4],X_train[:,:,4],X_train[:,:,5],X_train[:,:,6]],
                       Y_train,
                       batch_size=BATCH_SIZE * GPUS,
                       epochs=EPOCH,
                       validation_data=([X_validation[:,:,:4],X_validation[:,:,4],X_validation[:,:,5],X_validation[:,:,6]],
                                        Y_validation),
                       shuffle=True,
                       callbacks=_callbacks, verbose=1)

    Y_pred = parallel_model.predict([X_test[:,:,:4],X_test[:,:,4],X_test[:,:,5],X_test[:,:,6]])
    auc1 = metrics.roc_auc_score(Y_test[:, 0], Y_pred[:, 0])
    auc2 = metrics.roc_auc_score(Y_test[:, 1], Y_pred[:, 1])
    aupr1 = metrics.average_precision_score(Y_test[:, 0], Y_pred[:, 0])
    aupr2 = metrics.average_precision_score(Y_test[:, 1], Y_pred[:, 1])

    with open(os.path.join(save_dir, "auc.txt"), "w") as of:
        of.write("enhancer AUROC: %f\n" % auc1)
        of.write("enhancer AUPR: %f\n" % aupr1)
        of.write("silencer AUROC: %f\n" % auc2)
        of.write("silencer AUPR: %f\n" % aupr2)

def model_predict(data, model):
    print("prediction on test samples ...")
    y = model.predict(data, batch_size=1000, verbose=1)
    return y


def load_dataset(datafile):
    data = {}
    file = h5py.File(datafile, 'r')
    dict_group_load = file['data']
    dict_group_keys = dict_group_load.keys()
    for k in dict_group_keys:
        data[k] = dict_group_load[k][:]
    file.close()
    return data

def train_model(data, results_dir):
    model_file = WORK_DIR + "src/model.deeptact.7features.hdf5"
    model = load_model(model_file, compile=False, custom_objects={'attention': attention})

    if not os.path.exists(data):
        print("no data file" + data)
        exit()
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    data = load_dataset(data)
    run_model(data, model, results_dir)


if __name__ == "__main__":
    import sys

    data_file = sys.argv[1]
    results_dir = sys.argv[2]
    train_model(data_file, results_dir)
