import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from config import config

def getWindowedSplitData(dataset, waveIndexBegin, waveIndexEnding, tStepLeftShift=0, tStepRightShift=0, testSizePerc=0.2):
    """Function determines the size of the windows for the dataset
    param dataset: pd.Dataframe
        the dataset
    param waveIndexBegin: list
        contains the index for the begin of the window according to the LS1ON
    param waveIndexEnding: list
        contains the index for the ending of the window according to the LS1ON
    param tSstepLeftShift: int
        the number of data points that to the left of LS1ON that will be taken into account
    param tStepRightShift: int
        the number of data points that to the right of LS1ON that will be taken into account
    param expectedWaves: int or list (NOT USED)
        the number of waves per sequence, i.e., Bottle ColaHalb has 51 waves
    param testSizePerc : float between 0 and 1
        the percentage of the complete dataset that will be assign to the test/val dataset
    returns: np.array with the X_train, X_test, y_train, y_test"""

    num_classes = dataset["Bottle"].unique()
    # npDataSet = np.array(dataset["Data"]).reshape((len(dataset), -1))
    npDataSet = np.array(dataset.drop(dataset.columns[[0,3]], axis=1)).reshape((len(dataset), -1))
    batchedTrainData = []
    batchedLabels = []
    windowSize = -tStepLeftShift + tStepRightShift
    assert len(waveIndexBegin) == len(waveIndexEnding), "Lengh of indexes for begin and ending does not match"  #just as a checking
    #wie not use, perhaps in the future     
    for id, (wib, wie) in enumerate(zip(waveIndexBegin, waveIndexEnding)):
        batchedTrainData.append((npDataSet[wib+tStepLeftShift: wib+tStepRightShift, 0]))
        y_temp = npDataSet[wib+tStepLeftShift: wib+tStepRightShift, 1]
        if len(np.unique(y_temp)) == 1 and windowSize < 50:
        # y_temp.unique() == 1:
            batchedLabels.append(y_temp[0]) 
        else:
            raise ValueError("Window Size may overlap with not valid data points")

        
    
    
    X = np.array(batchedTrainData)
    # y = np.array(batchedLabels).reshape((len(batchedLabels), 1))
    y = np.array(batchedLabels).reshape((len(batchedLabels), ))
    
    # Implementing stratification for all the labels
    if config['valid_type'] == 'validNotSimplyRegression' or config['valid_type'] == 'validComplete':
        skf = StratifiedKFold(n_splits=2, random_state=config['seed'], shuffle=True)
        for train, test in skf.split(X,y):
            #lets take the first one
            index_train = train
            index_test = test
            break
        X_train = X[index_train]
        y_train = y[index_train]
        X_test = X[index_test]
        y_test = y[index_test]
        # # print(y_test[10])
        # # values, counts = np.unique(y_test, return_counts=True)    Helpful in printing
        # # print(counts)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSizePerc, random_state=config['seed'])

    return X_train, X_test, y_train, y_test

def MinMaxNormalization(X_train, X_test):
    """ Function performs a minmax normalization using sklearn,
    X_train is flatten and then normalization occurs, afterwards 
    the data is transform back to its original dimension
    param X_train: np array
        the X_train dataset with shape [windows, size_window]
    param X_test: np array
        the X_test dataset with shape [windows, size_window]
    returns: normalize X_train_norm and X_test_norm, also the mm object"""
    mm = MinMaxScaler()
    X_train_flatten = X_train.flatten().reshape(-1,1)
    X_test_flatten = X_test.flatten().reshape(-1,1)
    mm.fit(X_train_flatten)
    # Test to see if no weird dimmensions
    X_train_ss = mm.transform(X_train_flatten)
    X_train_ss = X_train_ss.reshape(len(X_train), -1)

    X_test_ss = mm.transform(X_test_flatten)
    X_test_ss = X_test_ss.reshape(len(X_test), -1)

    return X_train_ss, X_test_ss, mm