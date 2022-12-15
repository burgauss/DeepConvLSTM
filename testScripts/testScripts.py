import os
import errno

def main():
    path = "testScripts"
    newDir = os.path.join(path, "NewFolder", "file.txt")

    try:
        os.makedirs(newDir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    

if __name__ == "__main__":
    main()