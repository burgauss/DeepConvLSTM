import torch as nn
import torch
from config import config
import numpy as np
from misc.torchutils import seed_torch
import time
from dataLoader.DataLoader import myDataLoader
from preprocessing.preprocessing import *
from model.models import *
from model.train import *
from model.validate import *

# Path
pathAll = "C:/Users/juan.burgos/Desktop/JuanBurgos/04 Thesis/12_DataCollection/TrainSets/Combination/WeightLog_ALL.csv"
windowSizes = [ 102, 51, 103, 102, 102 , 101, 114]

seed_torch(config['seed'])          # allow random to be seeded for repeatibility


def main():
    log_date = time.strftime('%Y%m%d')
    log_timestamp = time.strftime('%H%M%S')

    dataLoader = myDataLoader(pathAll, True)
    dataset_pd, waveIndexBegin, waveIndexEnding = dataLoader.processData()
    X_train, X_valid, y_train, y_valid = getWindowedSplitData(dataset_pd, waveIndexBegin, waveIndexEnding, 
                            tStepLeftShift=0, tStepRightShift=15, testSizePerc=0.15)
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

    #Calling the model
    
    net = DeepConvLSTM_Simplified(config=config)
    

    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])

    ## Simplified Train
    # net = train_simplified(X_train_ss, y_train, X_test_ss, y_test,
    #    network=net, optimizer=opt, loss=loss, config=config, log_date=log_date,
    #    log_timestamp=log_timestamp)
    # torch.save(net.state_dict(), './model'+ log_date + log_timestamp +'.pth')

    ## Simplified train and validate
    # net = train_validate_simplified(X_train_ss, y_train, X_test_ss, y_test,
    #    network=net, optimizer=opt, loss=loss, config=config, log_date=log_date,
    #    log_timestamp=log_timestamp)
    # torch.save(net.state_dict(), './model'+ log_date + log_timestamp +'.pth')

    if config['valid_type'] == "split":
        net = DeepConvLSTM(config=config)
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        trained_net = train_valid_split(x_train_set = X_train_ss, y_train_set = y_train,
             x_valid_set = X_valid_ss, y_valid_set = y_valid, custom_net=net, custom_loss=loss, custom_opt=opt)
    else:
        net = DeepConvLSTM(config=config)
        loss = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        trained_net = train_validate_simplified(X_train_ss, y_train, X_valid_ss, y_valid,
        network=net, optimizer=opt, loss=loss, config=config, log_date=log_date,
        log_timestamp=log_timestamp)
        torch.save(trained_net.state_dict(), './model'+ log_date + log_timestamp +'.pth')
    
    # Validation Simplified
    #modelName = "model20221213082424.pth"
    #validation_simplified(modelName, X_valid_ss, y_valid, mm)

if __name__ == "__main__":
    main()