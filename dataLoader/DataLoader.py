import pandas as pd
import numpy as np

class myDataLoader():
    """Class upload the data from a csv file, moreover it determines the indexes of the windows
    at the beggining and at the ending
    params path: string
        an string with the path where the csv is to be found
    params info: boolean
        to allow printing of the info() and describe()
    returns a pandas dataframe , an np.array with index of wave being, np.array with index ending
    """
    def __init__(self, path, info):
        #self.path = path
        self.dataset = pd.read_csv(path, delimiter=";", header = None)
        self.info = info
        #self.dataset = self.dataset.drop(columns=[2,3,4,5,6,7,8], axis=1)
        
        #print(self.dataset.head(10))
    def processData(self):
        """ Method process the data according to the definition set in the WeightOfflineCalculation
        Args: None
        Returns: a pd.df with the processed data, an np.array with the index begin, an np array with the index ending
        """
        processedDataset = self.dataset.copy()
        processedDataset[10] = processedDataset[0].diff()
        waveLS1ON_1 = processedDataset.index[processedDataset[10]  == 1].to_numpy()
        waveLS1ON_5 = processedDataset.index[processedDataset[10]  == 5].to_numpy()
        waveLS1ON = np.sort(np.concatenate((waveLS1ON_1, waveLS1ON_5)))
        waveIndexBegin = processedDataset.index[processedDataset[10] == -1].to_numpy()  
        waveIndexEnding = processedDataset.index[processedDataset[10] == 2].to_numpy()
        # print("Windows start at" + str(waveIndexBegin)) # To print the actual index
        # print("Windows end at" + str(waveIndexEnding))
        print("Len of of index for start of Wave " + str(len(waveIndexBegin)))
        print("Len of of index for end of Wave " + str(len(waveIndexEnding)))
        ######################################################################
        # Droping the datasets
        processedDataset = processedDataset.drop(columns=[2,3,4,5,6,7,8], index=1)
        
        #Asssingning names to columns
        processedDataset.columns = ['Satus', 'Data', 'Bottle', 'Diff']
        
        if self.info == 1:
        # General information
            processedDataset.info()
            print(processedDataset.describe())

        return processedDataset, waveLS1ON, waveIndexEnding