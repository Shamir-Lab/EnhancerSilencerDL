from attention import attention
from keras.models import load_model, Model
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Concatenate
from keras.layers import Convolution1D, MaxPooling1D
from keras.regularizers import L1L2
from keras.constraints import max_norm

NUM_SEQ = 4
NUM_ENSEMBL = 20
RESIZED_LEN = 1000

def model_def():
    drop_rate = 0.5
    num_filters = 300
    filter_size = 19
    seq_input = Input(shape=(1000,4))
    meth_input = Input(shape=(1000, 1))
    h3k27ac_input = Input(shape=(1000,1))
    h3k4me1_input = Input(shape=(1000, 1))
    seq_conv = Convolution1D(num_filters,filter_size, name='seq_conv', activation = 'relu',kernel_regularizer=L1L2(l1=1e-8, l2=5e-7))(seq_input)
    seq_conv = MaxPooling1D(4,3)(seq_conv)
    meth_conv = Convolution1D(num_filters,filter_size, name='meth_conv', activation = 'relu',kernel_regularizer=L1L2(l1=1e-8, l2=5e-7))(meth_input)
    meth_conv = MaxPooling1D(4,3)(meth_conv)
    h3k27ac_conv = Convolution1D(num_filters,filter_size, name='h3k27ac_conv', activation = 'relu',kernel_regularizer=L1L2(l1=1e-8, l2=5e-7))(h3k27ac_input)
    h3k27ac_conv = MaxPooling1D(4,3)(h3k27ac_conv)
    h3k4me1_conv = Convolution1D(num_filters,filter_size, name='h3k4me1_conv', activation = 'relu',kernel_regularizer=L1L2(l1=1e-8, l2=5e-7))(h3k4me1_input)
    h3k4me1_conv = MaxPooling1D(4,3)(h3k4me1_conv)
    merged = Concatenate(axis=-1)([seq_conv,meth_conv,h3k27ac_conv,h3k4me1_conv])
    bn_1 = BatchNormalization()(merged)
    dp_1 = Dropout(drop_rate)(bn_1)
    lstm_1 = Bidirectional(LSTM(32,return_sequences = True))(dp_1)
    lstm_1 = attention()(lstm_1)
    bn_2 = BatchNormalization()(lstm_1)
    dp_2 = Dropout(drop_rate)(bn_2)
    output = Dense(1000,kernel_regularizer=L1L2(l1=1e-8, l2=5e-7))(dp_2)
    model = Model([seq_input,meth_input,h3k27ac_input,h3k4me1_input], output)
    return model

model = model_def()

model.save("models/model.for_reg.hdf5")

################################################################################################