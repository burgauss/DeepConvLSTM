import os
import errno
import sys
sys.path.append('C:/Users/juan.burgos/source/MasterArbeitSW/DeepConvLSTM/DeepConvLSTM/')
from misc.logging import Logger

def main():
    path = "testScripts"
    newDir = os.path.join(path, "NewFolder", "file.txt")

    try:
        os.makedirs(newDir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    #without a name
    sys.stdout = Logger(os.path.join(newDir, 'log.txt'))
    

if __name__ == "__main__":
    main()