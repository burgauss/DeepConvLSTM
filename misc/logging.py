##################################################
# Helper functions (logging-related)
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

from __future__ import absolute_import
import os
import sys
import torch
from torch.utils.data import DataLoader
from .osutils import mkdir_if_missing
from config import config
import pandas as pd
import numpy as np

class Logger(object):
    """
    Logger object used to redirect console output to textfile
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def logXtrainYtrain(logDir, X_train, y_train):
    """Following function creates a csv file containing one batch of the training dataset
    param logDir: strig
        the directory where the data will be saved
    param X_train: np.array
        Unormalized but batched X_train dataset
    param y_train: np.array
        Unormalized but batched y_train dataset
    returns: a csv file in the logDir
    """

    #Replicating the process in the training routine
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    trainLoader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True)
    
    samples_dict = {}
    

    for i, (x,y) in enumerate(trainLoader):
        XtrainLoader_batch = x
        ytrainLoader_batch = y
        if i == 1:
            break

    X_np = XtrainLoader_batch.numpy()
    y_np = ytrainLoader_batch.numpy()
    

    for i, (x,y) in enumerate(zip(X_np, y_np)):
        column_xy = np.concatenate((y,x), axis=None)
        samples_dict[str(i)] = column_xy
        # print("sample= ", x)
        # print("first sample label= ", y)
    samples_df = pd.DataFrame.from_dict(samples_dict)
    logDir = os.path.join(logDir, 'log.csv')
    samples_df.to_csv(logDir, sep=';', decimal=",")
    #print(samples_df)