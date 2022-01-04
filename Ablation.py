import os.path
import sys
import time
import pickle
import h5py
import os
import numpy as np
import warnings
import copy
import math
import random
from sklearn.utils import shuffle

from scipy import stats
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
from sklearn.preprocessing import StandardScaler


AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

def reset_keras(classifier1, classifier2):
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier1 # this is from global space - change this as you need
        del classifier2
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def labelize(plaintexts, keys):
    return AES_Sbox[plaintexts ^ keys]

#### ASCAD helper to load profiling and attack data (traces and labels)
def addGussianNoise(traces, noise_level):
    print('Add Gussian noise: ', noise_level)
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        for trace in range(len(traces)):
            if(trace % 5000 == 0):
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces

# Loads the profiling and attack datasets from the ASCAD database
def load_ascad(root, dataset, filename, noise_level, load_metadata=False):
    try:
        in_file = h5py.File(root + dataset + '/Base_desync0.h5', "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ...")
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.load(root + dataset + '/' + filename)[:50000]
    #X_profiling = addGussianNoise(X_profiling, noise_level)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])[:50000]
    # Load attacking traces
    X_attack = np.load(root + dataset + '/' + filename)[50000:60000]
    #X_attack = addGussianNoise(X_attack, noise_level)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])[:10000]
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling[:50000], Y_profiling[:50000]), (X_attack[:50000], Y_attack[:50000]), (in_file['Profiling_traces/metadata']['plaintext'][:50000], in_file['Attack_traces/metadata']['plaintext'][:50000])

# Loads the profiling and attack datasets from the ASCAD database
def load_ascad_r(root, dataset, filename, noise_level, load_metadata=False):
    try:
        in_file = h5py.File(root + dataset + '/ascad-variable_GNoise_0.h5', "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ...")
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.load(root + dataset + '/' + filename)[:50000]
    #X_profiling = addGussianNoise(X_profiling, noise_level)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])[:50000]
    # Load attacking traces
    X_attack = np.load(root + dataset + '/' + filename)[50000:60000]
    #X_attack = addGussianNoise(X_attack, noise_level)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])[:10000]
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling[:50000], Y_profiling[:50000]), (X_attack[:50000], Y_attack[:50000]), (in_file['Profiling_traces/metadata']['plaintext'][:50000], in_file['Attack_traces/metadata']['plaintext'][:50000])


def load_chipwhisperer(database_file, noise_level=0, load_metadata=False):
    #traces = np.load(database_file + '/traces_noisy.npy')[:10000]
    
    X = np.load(database_file + '/traces.npy')[:10000]
    X = addGussianNoise(X, noise_level)
    #np.save(database_file + '/traces_noisy', traces)
    P = np.load(database_file + '/plain.npy')[:, 1]
    K = np.load(database_file + '/key.npy')[:, 1]
    Y = labelize(P, K)
    plaintexts = np.zeros((len(P), 8))
    plaintexts[:, 2] = P
    if load_metadata == False:
        return (X[:8000], Y[:8000]), (X[8000:], Y[8000:])
    else:
        return (X[:8000], Y[:8000]), (X[8000:], Y[8000:]), (plaintexts[:8000], plaintexts[8000:])

def show(plot):
    plt.plot(plot)
    plt.show()

def plot_all_GE(container, real_key):
    if len(container) != 256:
        container = np.transpose(container)
        #np.save(root+"Avg_"+name, container)

    idx = 0
    for GE in container:
        if idx == real_key:
            plt.plot(GE,  color='red', zorder=500)
        else:
            plt.plot(GE,  color='grey')
        idx += 1
    #plt.savefig(root+'GE_all_'+name+'.png', format='png')
    plt.show()

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[s] for s in data]

# Compute the position of the key hypothesis key amongst the hypotheses
def rk_key(rank_array, key):
    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])
    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256)
    else:
        return np.float32(final_rank)

# Compute the evolution of rank
def rank_compute(prediction, att_plt, byte, output_rank):
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    prediction = np.log(prediction+1e-40)
    rank_evol = np.full(nb_traces,255)

    for i in range(nb_traces):
        for k in range(256):
            # Computes the hypothesis values
            if leakage == 'ID':
                key_log_prob[k] += prediction[i, AES_Sbox[k ^ int(att_plt[i, byte])]]
            else:
                key_log_prob[k] += prediction[i, hw[AES_Sbox[k ^ int(att_plt[i, byte])]]]
        rank_evol[i] = rk_key(key_log_prob, correct_key)

    if output_rank:
        return rank_evol
    else:
        return key_log_prob


def perform_attacks(nb_traces, predictions, plt_attack, nb_attacks=1, byte=2, shuffle=True, output_rank=False):
    (nb_total, nb_hyp) = predictions.shape
    all_rk_evol = np.zeros((nb_attacks, nb_traces))

    for i in range(nb_attacks):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]

        key_log_prob = rank_compute(att_pred, att_plt, byte, output_rank)
        if output_rank:
            all_rk_evol[i] = key_log_prob

    if output_rank:
        return np.mean(all_rk_evol,axis=0)  
    else:
        return np.float32(key_log_prob)


def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

### CNN network
def cnn_methodology(input_size=700, classes=256, learning_rate=5e-3):
        # Designing input layer
        input_shape = (input_size,1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        
        x = Flatten(name='flatten')(x)

        # Classification layer
        x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
        x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
        
        # Logits layer              
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create model
        inputs = img_input
        model = Model(inputs, x, name='ascad')
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        return model

def cnn_best(length, classes=256):
    # From VGG16 design
    img_input = Input(shape=(length, 1))
    filter_array = [128, 256, 512, 512]
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same')(img_input)
    x = AveragePooling1D(2, strides=2)(x)

    for i in range(len(filter_array)):
        x = Conv1D(filter_array[i], 11, activation='relu', padding='same')(x)
        x = AveragePooling1D(2, strides=2)(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    # Create model.
    model = Model(img_input, x)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.00001), metrics=['accuracy'])
    model.summary()
    return model

def mlp_best(length, classes=256, lr=0.00001, node=200, layer_nb=6):
    img_input = Input(shape=(length, ))
    x = Dense(200, activation='relu')(img_input)
    #skip = x
    x = Dense(200, activation='relu')(x)
    #x = Add()([x,skip])
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def mlp_autosca(length, classes=256, lr=0.00001):
    img_input = Input(shape=(length, ))
    x = Dense(200, activation='relu')(img_input)
    #skip = x
    x = Dense(200, activation='relu')(x)
    #x = Add()([x,skip])
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def mlp_rev(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)
    x = Flatten(name='flatten')(x)

    x = Dense(10, activation='selu', name='fc1')(x)
    x = Dense(10, activation='selu', name='fc2')(x)          
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_0')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def mlp_chipw(length, classes=256, lr=0.00001):
    model = Sequential()
    model.add(Dense(3, input_dim=length, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def get_model_weight_bias(model):
    return [np.array(w) for w in model.get_weights()]

def calculate_weight_after_ablation(model, ablation_layer, ablation_neuron):
    weight_bias = get_model_weight_bias(model)

    if len(ablation_neuron) == 0:
      print('No ablation neuron being specified, skip abalation')
      return weight_bias

    if attack_model == 'MLP':
        # delete incoming weight and bias
        weight_bias[(ablation_layer)*2] = np.delete(weight_bias[(ablation_layer)*2], ablation_neuron, axis=1)
        weight_bias[(ablation_layer)*2+1] = np.delete(weight_bias[(ablation_layer)*2+1], ablation_neuron)
        # delete outcoming weight (bias belongs to next layer so don't delelte)
        weight_bias[(ablation_layer+1)*2] = np.delete(weight_bias[(ablation_layer+1)*2], ablation_neuron, axis=0)
    else:
        # delete incoming weight and bias
        weight_bias[(ablation_layer)*2] = np.delete(weight_bias[(ablation_layer)*2], ablation_neuron, axis=2)
        print(np.shape(weight_bias[(ablation_layer)*2]))
        weight_bias[(ablation_layer)*2+1] = np.delete(weight_bias[(ablation_layer)*2+1], ablation_neuron)
        # delete outcoming weight (bias belongs to next layer so don't delelte)
        weight_bias[(ablation_layer+1)*2] = np.delete(weight_bias[(ablation_layer+1)*2], ablation_neuron, axis=1)

    return weight_bias

def calculate_weight_variation(weight_bias_before_ablation, weight_bias_after_ablation):
    w_diff_array = []
    b_diff_array = []
    #weight_bias_before_ablation = calculate_weight_after_ablation(model_before_ablation, ablation_layer, ablation_neuron)
    # weight_bias_before_ablation = weight_bias_after_ablation
    # weight_bias_after_ablation = weight_bias_after_recovery

    if attack_model == 'MLP' or attack_model == 'AutoSCA_MLP':
        for i in range(int(len(weight_bias_before_ablation)/2)):
            w_diff = np.mean(abs(weight_bias_before_ablation[i*2]-weight_bias_after_ablation[i*2]), axis=1)
            b_diff = abs(weight_bias_before_ablation[i*2+1]-weight_bias_after_ablation[i*2+1])
            w_diff_array.append(w_diff)
            b_diff_array.append(b_diff)
    else:
        for i in range(int(len(weight_bias_before_ablation)/2)):
            if len(np.shape(weight_bias_before_ablation[i*2])) == 3:
                w_diff = []
                for j in range(weight_bias_before_ablation[i*2].shape[2]):
                    w_diff.append(np.mean(abs(weight_bias_before_ablation[i*2][:,:,j]-weight_bias_after_ablation[i*2][:,:,j])))
                b_diff = abs(weight_bias_before_ablation[i*2+1]-weight_bias_after_ablation[i*2+1])
                w_diff_array.append(w_diff)
                b_diff_array.append(b_diff)
            elif len(np.shape(weight_bias_before_ablation[i*2])) == 2:
                w_diff = np.mean(abs(weight_bias_before_ablation[i*2]-weight_bias_after_ablation[i*2]), axis=1)
                b_diff = abs(weight_bias_before_ablation[i*2+1]-weight_bias_after_ablation[i*2+1])
                w_diff_array.append(w_diff)
                b_diff_array.append(b_diff)
    return np.array(w_diff_array), np.array(b_diff_array)

#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, saved_models_dir, epochs=150, batch_size=100, save_model=False):
    history = model.fit(x=X_profiling, y=Y_profiling, validation_data=(X_test, Y_test), batch_size=batch_size, verbose = 2, epochs=epochs)
    if save_model:
        model.save(saved_models_dir)
    return history

def create_zero_matrix(template):
    container = []
    for item in template:
        container.append(np.zeros_like(item))
    return np.array(container)


if __name__ == "__main__":
    root = ''

    saved_models_dir = ''
    result_dir = ''

    dataset = model_name = ''
    attack_model_array = [''] #['MLP_SIMP','MLP_DIFF','CNN_SIMP','CNN_DIFF']
    leakage_array = ['HW','ID'] #['HW','ID']

    recovery_epoch = 
    ablation_times = 

    nb_attacks = 
    
    countermeasures = str(sys.argv[1]) #Desync,Clock_Range,GNoise

    if countermeasures == 'Desync':
        noise = '' # Desync level
    else:
        noise = '' # Gaussian Noise Level

    print('load datasets')

    # Load the profiling traces
    for attack_model in attack_model_array:
        for leakage in leakage_array:
            if dataset == 'ASCAD':
                correct_key = 224
                nb_traces_attacks = 5000
                #data_root = 'ASCAD/Base_desync0.h5'
                data_root = 'ASCAD_' + countermeasures + '_' +str(noise) + '.npy'
                (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(root, dataset, data_root, noise, load_metadata=True)
            elif dataset == 'ASCAD_rand':
                correct_key = 34
                nb_traces_attacks = 5000
                data_root = 'ascad-variable_' + countermeasures + '_' +str(noise) + '.npy'
                (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad_r(root, dataset, data_root, noise, load_metadata=True)
            elif dataset == 'CHIPW':
                correct_key = 126
                nb_traces_attacks = 2000
                data_root = 'Chipwhisperer/'
                (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_chipwhisperer(root + data_root, noise, load_metadata=True)

            if leakage == 'ID':
                print('ID leakage model')
                classes = 256
            else:
                print('HW leakage model')
                classes = 9
                Y_profiling = calculate_HW(Y_profiling)
                Y_attack = calculate_HW(Y_attack)

            scaler = StandardScaler()
            X_profiling = scaler.fit_transform(X_profiling)
            X_attack = scaler.transform(X_attack)  

            if attack_model == 'MLP_SIMP':# HW:lr_0.0001/epoch_50 ID:lr_0.00003/epoch_200 
                X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1],1))
                X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1],1))
                if leakage == 'ID':
                    lr=0.005
                    nb_epochs = 100
                else:
                    lr=0.005
                    nb_epochs = 100
                batch_size = 50
                model_before_ablation = mlp_rev(input_size=len(X_profiling[0]),learning_rate=lr,classes=classes)
                optimizer=RMSprop(lr=lr)
                ablation_layers_range = [-1,3,4]
                neuron=[10,10]
                ablation_ratio = np.array([0.1, 0.5, 0.9])

            elif attack_model == 'MLP_DIFF': # HW:lr_0.0001/epoch_100 ID:lr_0.00003/epoch_200 
                X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
                X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
                if leakage == 'ID':
                    lr=0.0005
                    nb_epochs = 100
                else:
                    lr=0.0005
                    nb_epochs = 100
                batch_size = 200
                model_before_ablation = mlp_best(len(X_profiling[0]), classes, lr=lr)
                optimizer=RMSprop(lr=lr)
                ablation_layers_range = [1,2,3,4,5]
                neuron=[200,200,200,200,200]
                ablation_ratio = np.array([0.1, 0.5, 0.9])

            elif attack_model == 'MLP_VERY_DIFF': # HW:lr_0.0001/epoch_100 ID:lr_0.00003/epoch_200 
                X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
                X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
                if leakage == 'ID':
                    lr=0.0005
                    nb_epochs = 100
                else:
                    lr=0.0005
                    nb_epochs = 100
                batch_size = 200
                model_before_ablation = mlp_autosca(len(X_profiling[0]), classes, lr=lr)
                optimizer=RMSprop(lr=lr)
                ablation_layers_range = [-1,1,2,3,4,5,6,7,8]
                neuron=[200,200,200,200,200,200,200,200]
                ablation_ratio = np.array([0.1, 0.5, 0.9])

            elif attack_model == 'CNN_SIMP':
                X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
                X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
                model_before_ablation = cnn_methodology(len(X_profiling[0]), classes)
                # training failed
                if leakage == 'ID':
                    lr=1e-6
                    nb_epochs = 200
                else:
                    lr=5e-4
                    nb_epochs = 100
                batch_size = 50
                optimizer=Adam(lr=lr)
                ablation_layers_range = [-1,1,5,6]
                neuron=[4,10,10]
                ablation_ratio = np.array([0.1, 0.5, 0.9])

            elif attack_model == 'CNN_DIFF':
                X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
                X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
                model_before_ablation = cnn_best(len(X_profiling[0]), classes)
                nb_epochs = 75
                lr=0.0001
                batch_size = 256
                optimizer=RMSprop(lr=lr)
                ablation_layers_range = [-1,1,3,5,7,9,12,13]
                neuron=[64,128,256,512,512,1024,1024]
                ablation_ratio = np.array([0.1, 0.5, 0.9])

            Y_profiling = to_categorical(Y_profiling, num_classes=classes)
            Y_attack = to_categorical(Y_attack, num_classes=classes)

            # Train model
            print('Train model...')
            model_before_ablation = load_model(saved_models_dir + model_name + '_' + attack_model + '_' + leakage + "_before_ablation.h5")
            #train_model(X_profiling[:45000], Y_profiling[:45000], X_attack[:1000], Y_attack[:1000], model_before_ablation, saved_models_dir + model_name + '_' + attack_model + '_' + leakage + "_before_ablation_Lm.h5", epochs=nb_epochs, batch_size=batch_size, save_model=True) 

            # Get weight and bias before ablation
            print('Get weight and bias before ablation...')
            weight_bias_before_ablation = get_model_weight_bias(model_before_ablation)

            # # Attack data before ablation
            predictions = model_before_ablation.predict(X_attack)
            avg_rank = np.array(perform_attacks(nb_traces_attacks, predictions, plt_attack, nb_attacks=nb_attacks, byte=2, shuffle=True, output_rank=True))
            plt.plot(avg_rank)
            plt.show()
            print('GE smaller that 1:', np.argmax(avg_rank < 1))
            print('GE smaller that 5:', np.argmax(avg_rank < 5))

            # Ablate model
            print('Ablate model...')
            for ratio in ablation_ratio:
                print('====== Ablation ratio: ', ratio)
                w_difference_nomean_array = []
                b_difference_nomean_array = []
                w_difference_mean_array = []
                b_difference_mean_array = []
                avg_rank_array_before_recovery = []
                avg_rank_array_after_recovery = []
                layer_index = -1
                for ablation_layer in ablation_layers_range:
                    print('======= Ablating on layer', ablation_layer,'=======')
                    avg_rank_before = 0
                    avg_rank_after = 0
                    w_difference_array = []
                    b_difference_array = []
                    for j in range(ablation_times):
                        print('Ablation times: ', j)
                        #model_before_ablation = load_model(saved_models_dir + model_name + '_' + attack_model + '_' + leakage + "_before_ablation.h5")
                        model_before_ablation.set_weights(weight_bias_before_ablation)
                        if ablation_layer == -1:
                            # reference (no ablation)
                            model_after_ablation = model_before_ablation
                        else:
                            # Ablation
                            model_after_ablation = delete_channels(model_before_ablation, model_before_ablation.layers[ablation_layer], shuffle(np.array(range(neuron[layer_index])))[:int(neuron[layer_index]*ratio)])

                        model_after_ablation.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                        model_after_ablation.summary()
                        weight_bias_after_ablation = get_model_weight_bias(model_after_ablation)

                        # Attack data after ablation
                        predictions = model_after_ablation.predict(X_attack)
                        avg_rank_before += np.array(perform_attacks(nb_traces_attacks, predictions, plt_attack, nb_attacks=nb_attacks, byte=2, shuffle=True, output_rank=True))

                        print('GE smaller that 1:', np.argmax(avg_rank_before < 1))
                        print('GE smaller that 5:', np.argmax(avg_rank_before < 5))

                        # Recovery training
                        print('Recovery training...')
                        train_model(X_profiling[:45000], Y_profiling[:45000], X_attack[:1000], Y_attack[:1000], model_after_ablation, saved_models_dir + model_name  + '_' + attack_model + '_' + leakage + "_after_ablation_Lm.h5", epochs=recovery_epoch, batch_size=batch_size) 
                        weight_bias_after_recovery = get_model_weight_bias(model_after_ablation)
                        # Attack data after recovery training
                        predictions = model_after_ablation.predict(X_attack)
                        avg_rank_after += np.array(perform_attacks(nb_traces_attacks, predictions, plt_attack, nb_attacks=nb_attacks, byte=2, shuffle=True, output_rank=True))

                        print('GE smaller that 1:', np.argmax(avg_rank_after < 1))
                        print('GE smaller that 5:', np.argmax(avg_rank_after < 5))
                        
                        # Calculate weight difference
                        print('calculate weight difference...')
                        w_diff, b_diff = calculate_weight_variation(weight_bias_after_ablation, weight_bias_after_recovery)
                        w_difference_array.append(w_diff)
                        b_difference_array.append(b_diff)
                        #del 
                        K.clear_session()
                    
                    #reset_keras(model_before_ablation, model_after_ablation)

                    avg_rank_array_before_recovery.append(avg_rank_before/ablation_times)
                    avg_rank_array_after_recovery.append(avg_rank_after/ablation_times)
                    weight_diff_mean = np.zeros((len(w_difference_array[0]),))
                    bias_diff_mean = np.zeros((len(b_difference_array[0]),))
                    weight_diff = create_zero_matrix(w_difference_array[0])
                    bias_diff = create_zero_matrix(b_difference_array[0])

                    for x in range(len(w_difference_array)):
                        for y in range(len(w_difference_array[0])):
                            weight_diff[y] += w_difference_array[x][y]
                            bias_diff[y] += b_difference_array[x][y]
                            weight_diff_mean[y] += np.mean(w_difference_array[x][y])
                            bias_diff_mean[y] += np.mean(b_difference_array[x][y])
                    w_difference_nomean_array.append(weight_diff/ablation_times)
                    b_difference_nomean_array.append(bias_diff/ablation_times)
                    w_difference_mean_array.append(weight_diff_mean/ablation_times)
                    b_difference_mean_array.append(bias_diff_mean/ablation_times)

                    layer_index += 1

                    np.save(result_dir + countermeasures + str(noise) + '_GE_before_recovery_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) +'.npy', np.array(avg_rank_array_before_recovery))
                    np.save(result_dir + countermeasures + str(noise) + '_GE_after_recovery_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.npy', np.array(avg_rank_array_after_recovery))
                    np.save(result_dir + countermeasures + str(noise) + '_w_diff_mean_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.npy', np.array(w_difference_mean_array))
                    np.save(result_dir + countermeasures + str(noise) + '_b_diff_mean_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.npy', np.array(b_difference_mean_array))
                    np.save(result_dir + countermeasures + str(noise) + '_w_diff_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.npy', np.array(w_difference_nomean_array))
                    np.save(result_dir + countermeasures + str(noise) + '_b_diff_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.npy', np.array(b_difference_nomean_array))
                    
                # Plot GE before recovery per layer results
                for i in range(len(avg_rank_array_before_recovery)):
                    if i == 0:
                        plt.plot(avg_rank_array_before_recovery[i], label='GE_ref')
                    else:
                        plt.plot(avg_rank_array_before_recovery[i], label='GE_bef_L' + str(i-1))
                plt.legend(loc=1)
                plt.savefig(result_dir + countermeasures + str(noise) + '_GE_before_recovery_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.png')
                #plt.show()
                plt.clf()

                # Plot GE after recovery per layer results
                for i in range(len(avg_rank_array_after_recovery)):
                    if i == 0:
                        plt.plot(avg_rank_array_after_recovery[i], label='GE_ref')
                    else:
                        plt.plot(avg_rank_array_after_recovery[i], label='GE_aft_L' + str(i-1))
                plt.legend(loc=1)
                plt.savefig(result_dir + countermeasures + str(noise) + '_GE_after_recovery_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.png')
                #plt.show()
                plt.clf()

                # Plot w_diff_mean per layer results
                for i in range(len(w_difference_mean_array)):
                    if i == 0:
                        plt.plot(w_difference_mean_array[i], label='w_ref')
                    else:
                        plt.plot(w_difference_mean_array[i], label='w_L' + str(i-1))
                plt.legend(loc=2)
                plt.savefig(result_dir + countermeasures + str(noise) + '_w_diff_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.png')
                #plt.show()
                plt.clf()

                # Plot b_diff_mean per layer results
                for i in range(len(b_difference_mean_array)):
                    if i == 0:
                        plt.plot(b_difference_mean_array[i], label='b_ref')
                    else:
                        plt.plot(b_difference_mean_array[i], label='b_L' + str(i-1))
                plt.legend(loc=2)
                plt.savefig(result_dir + countermeasures + str(noise) + '_b_diff_' + model_name + '_' + attack_model + '_' + leakage + '_AbRatio_'+ str(ratio) + '.png')
                #plt.show()
                plt.clf()

                if ablation_layer == -1:
                    break
    print('Done!')

