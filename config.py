import os 
class CONFIG:
    PATH = os.getcwd()
    DATA_PATH = os.path.join(PATH, "dataset/train.csv")
    STORE_FILE_PATH = os.path.join(PATH, "dataset")

    SAVE_MODEL_PATH = os.path.join(PATH, "model/")
