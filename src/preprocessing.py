import pandas as pd
from config import CONFIG
import string
from sklearn.model_selection import train_test_split
class PREP:
    def __init__(self) -> None:
        df = pd.read_csv(CONFIG.DATA_PATH)
        df["sentencePrep"] = df["sms"].apply(self.removePunctuation)
        df_new = df[["sentencePrep", "label"]]
        train, test = train_test_split(df_new, test_size=0.3, stratify=df_new["label"])
        train.to_csv(CONFIG.STORE_FILE_PATH+"/train_refactor.csv", index=False)
        test.to_csv(CONFIG.STORE_FILE_PATH+"/test_refactor.csv", index=False)

    def removePunctuation(self, sentence):
        return sentence.translate(str.maketrans("", "", string.punctuation)).lower()

PREP()