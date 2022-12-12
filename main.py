import torch as nn
import torch
from config import config
import numpy as np
from misc.torchutils import seed_torch
import time
from dataLoader.DataLoader import DataLoader
from preprocessing.preprocessing import *


# Path
pathAll = "C:/Users/juan.burgos/Desktop/JuanBurgos/04 Thesis/12_DataCollection/TrainSets/Combination/WeightLog_ALL.csv"
windowSizes = [ 102, 51, 103, 102, 102 , 101, 114]

seed_torch(config['seed'])          # allow random to be seeded for repeatibility



def main():
    log_date = time.strftime('%Y%m%d')
    log_timestamp = time.strftime('%H%M%S')

    dataLoader = DataLoader(pathAll, info=1)
    dataset_pd, waveIndexBegin, waveIndexEnding = dataLoader.processData()
    X_train, X_test, y_train, y_test = getWindowedSplitData(dataset_pd, waveIndexBegin, waveIndexEnding, 
                            tStepLeftShift=0, tStepRightShift=15, expectedWavesSizes=windowSizes)
    X_train_ss, X_test_ss, mm = MinMaxNormalization(X_train, X_test)             # Rescaling

    print("X_train shape: ", X_train_ss.shape, "X_test_shape", X_test_ss.shape)

    #Adding an extradimension for Pytorch
    X_train_ss = X_train_ss.reshape((X_train_ss.shape[0], X_train_ss.shape[1], 1))
    X_test_ss = X_test_ss.reshape((X_test_ss.shape[0], X_test_ss.shape[1], 1))
    
    print("X_train new shape: ", X_train_ss.shape, "y_train shape", y_train.shape)
    # Converting data for GPU compatibality
    X_train_ss, y_train = X_train_ss.astype(np.float32), y_train.astype(np.uint8)
    X_test_ss, y_test = X_test_ss.astype(np.float32), y_test.astype(np.uint8)

    #Adding two new parameters according to the shape of the datasets
    config['window_size'] = X_train_ss.shape[1]
    config['nb_channels'] = X_train_ss.shape[2]


if __name__ == "__main__":
    main()