import torch as nn
import torch
from config import config
import numpy as np
import os
import sys
from misc.logging import Logger, logXtrainYtrain
from misc.osutils import mkdir_if_missing
from misc.torchutils import seed_torch
import time
from dataLoader.DataLoader import myDataLoader
from preprocessing.preprocessing import *
from model.models import *
from model.train import *
from model.validate import *

# Path
pathAll = "C:/Users/juan.burgos/Desktop/JuanBurgos/04 Thesis/12_DataCollection/TrainSets/Combination/WeightLog_ALL.csv"
pathAll_regression = "C:/Users/juan.burgos/Desktop/JuanBurgos/04 Thesis/12_DataCollection/TrainSets/Combination/WeightLog_ALL_regression.csv"
windowSizes = [ 102, 51, 103, 102, 102 , 101, 114]

seed_torch(config['seed'])          # allow random to be seeded for repeatibility


def main():
    log_date = time.strftime('%Y%m%d')
    log_timestamp = time.strftime('%H%M%S')


    dataLoader = myDataLoader(pathAll, True)
    dataset_pd, indexes_LS1ON, indexes_LS2ON = dataLoader.processData()
    X_train, X_valid, y_train, y_valid = getWindowedSplitData(dataset_pd, indexes_LS1ON, indexes_LS2ON, 
                            tStepLeftShift=0, tStepRightShift=45, testSizePerc=0.15)
    X_train_ss, X_valid_ss, mm = MinMaxNormalization(X_train, X_valid)             # Rescaling

    print("X_train shape: ", X_train_ss.shape, "X_test_shape", X_valid_ss.shape)

    #Adding an extradimension for Pytorch
    X_train_ss = X_train_ss.reshape((X_train_ss.shape[0], X_train_ss.shape[1], 1))
    X_valid_ss = X_valid_ss.reshape((X_valid_ss.shape[0], X_valid_ss.shape[1], 1))
    
    print("X_train new shape: ", X_train_ss.shape, "y_train shape", y_train.shape)
    # Converting data for GPU compatibality
    X_train_ss, y_train = X_train_ss.astype(np.float32), y_train.astype(np.uint8)
    X_valid_ss, y_valid = X_valid_ss.astype(np.float32), y_valid.astype(np.uint8)

    #Adding two new parameters according to the shape of the datasets
    config['window_size'] = X_train_ss.shape[1]
    config['nb_channels'] = X_train_ss.shape[2]


    if config['valid_type'] == "split":
        net = DeepConvLSTM(config=config)
        print(net)
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        trained_net = train_valid_split(x_train_set = X_train_ss, y_train_set = y_train,
             x_valid_set = X_valid_ss, y_valid_set = y_valid, custom_net=net, custom_loss=loss, custom_opt=opt)
        torch.save(trained_net.state_dict(), './model'+ log_date + log_timestamp +'.pth')
    elif config['valid_type'] == 'trainValidSimply':
        net = DeepConvLSTM_Simplified(config=config)
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        trained_net = train_validate_simplified(X_train_ss, y_train, X_valid_ss, y_valid,
        network=net, optimizer=opt, loss=loss, log_date=log_date,
        log_timestamp=log_timestamp)
        torch.save(trained_net.state_dict(), './modelSimply'+ log_date + log_timestamp +'.pth')
    elif config['valid_type'] == 'validSimply':
        #Validation Simplified
        modelName = "modelSimply20221214161110.pth"
        validation_simplified(modelName, X_valid_ss, y_valid, mm)
    elif config['valid_type'] == 'validNotSimply':
        modelName = "model20221215114829.pth"
        validation(modelName, X_valid_ss, y_valid, mm)
    elif config['valid_type'] == "logoutInputBatched":
        #Implementing the Logger
        log_dir = os.path.join('logs', log_date, log_timestamp)
        mkdir_if_missing(log_dir)
        logXtrainYtrain(log_dir, X_train, y_train)  	#Log one sample for thesis purposes

def main_regression():
    log_date = time.strftime('%Y%m%d')
    log_timestamp = time.strftime('%H%M%S')

    print(np.__version__)
    dataLoader = myDataLoader(pathAll_regression, True)
    dataset_pd, indexes_LS1ON, indexes_LS2ON, indexes_LS1OFF = dataLoader.processData()
    #Training for iniital window
    X_train, X_valid, y_train, y_valid = getWindowedSplitData(dataset_pd, indexes_LS1ON, indexes_LS2ON, 
                            tStepLeftShift=0, tStepRightShift=45, testSizePerc=0.20)
    #Training for PIB window
    # X_train, X_valid, y_train, y_valid = getWindowedSplitData(dataset_pd, indexes_LS1OFF, indexes_LS2ON, 
    #                         tStepLeftShift=-10, tStepRightShift=10, testSizePerc=0.20)
    # Observing the distribution of data from y_valid
    value, counts = np.unique(y_valid, return_counts=True)
    print("values: ", value)
    print("counts: ", counts)
    X_train_ss, X_valid_ss, x_mm = MinMaxNormalization(X_train, X_valid)             # Rescaling
    y_train_ss, y_valid_ss, y_mm = MinMaxNormalization(y_train, y_valid)
 
    print("X_train shape: ", X_train_ss.shape, "X_test_shape", X_valid_ss.shape)
    print("y_train shape: ", y_train_ss.shape, "y_test_shape", y_valid_ss.shape)
    #Adding an extradimension for Pytorch
    X_train_ss = X_train_ss.reshape((X_train_ss.shape[0], X_train_ss.shape[1], 1))
    X_valid_ss = X_valid_ss.reshape((X_valid_ss.shape[0], X_valid_ss.shape[1], 1))
    
    print("X_train new shape: ", X_train_ss.shape, "y_train shape", y_train.shape)
    # Converting data for GPU compatibality
    X_train_ss, y_train_ss = X_train_ss.astype(np.float32), y_train_ss.astype(np.float32)
    X_valid_ss, y_valid_ss = X_valid_ss.astype(np.float32), y_valid_ss.astype(np.float32)

    #Adding two new parameters according to the shape of the datasets
    config['window_size'] = X_train_ss.shape[1]
    config['nb_channels'] = X_train_ss.shape[2]

    if config['valid_type'] == "split":
        net = DeepConvLSTM_regression(config=config)
        loss = torch.nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        trained_net, checkpoint, val_output, train_output, best_fp, best_fn, best_precision, counter  = train_valid_split(x_train_set = X_train_ss, y_train_set = y_train_ss,
             x_valid_set = X_valid_ss, y_valid_set = y_valid_ss, custom_net=net, custom_loss=loss, custom_opt=opt, y_scaler=y_mm)
        print("final_valoutput")
        print(val_output)
        torch.save(trained_net.state_dict(), './model_regression'+ log_date + log_timestamp +'.pth')
    elif config['valid_type'] == 'validNotSimplyRegression':
        modelName = "model_regression20221219120539.pth"
        validation_regression(modelName, X_valid_ss, y_valid_ss, x_mm, y_mm)

if __name__ == "__main__":
    if config['DL_mode'] == 'classification':
        main()
    elif config['DL_mode'] == 'regression':
        main_regression()