import torch
import pandas as pd
from config import CONFIG
import numpy as np
import json
import os 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score
# import evaluate

class dataset:
    def __init__(self, tokenizer, df) -> None:
        self.tokenizer = tokenizer
        self.df = df 



    def __getitem__(self, idx):
        input_ids = self.tokenizer["input_ids"][idx]
        attention_mask = self.tokenizer["attention_mask"][idx]
        label = self.df["label"].iloc[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }


    def __len__(self):
        return self.df.shape[0]


class training:
    def __init__(self) -> None:
        # self.f1_score = evaluate.load("f1")
        # self.preicision= evaluate.load("precision")
        # self.recall = evaluate.load("recall")
        train_file = CONFIG.STORE_FILE_PATH+"/train_refactor.csv"
        test_file  = CONFIG.STORE_FILE_PATH+"/test_refactor.csv"
        self.df_train = pd.read_csv(train_file)[:100]
        self.df_test  = pd.read_csv(test_file)[:50]
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model     = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

        # tokenized train 
       
        tokenizedTrain = self.tokenizer(self.df_train["sentencePrep"].values.tolist(), max_length = 256, padding="max_length",truncation=True,  return_tensors="pt")
        # tokenized test 
        tokenizedTest = self.tokenizer(self.df_test["sentencePrep"].values.tolist(), max_length=256, padding="max_length", truncation=True,  return_tensors="pt")

        # train and test dataset
        trainDataset = dataset(tokenizedTrain, self.df_train)
        testDataset = dataset(tokenizedTest, self.df_test)

        # train and test loader
        self.trainloader = DataLoader(trainDataset, batch_size=32, shuffle=True)
        self.testloader  = DataLoader(testDataset, batch_size=32, shuffle=False)

        # parameters 
        self.EPOCH = 5
        self.val = 2

        # define the optimizer
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.metrics_prc = []
    def trainModel(self):
        for i in tqdm(range(self.EPOCH)):
            for data in tqdm(self.trainloader):
                output = self.model(**data)
                loss = output.loss
                logits = output.logits
                probability = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(probability, dim=-1)
                loss.backward()
                self.optimizers.step()
                self.optimizers.zero_grad()

            
            if i % self.val == 0 and i != 0:
                self.model.eval()
                with torch.no_grad():
                    for data in tqdm(self.testloader):
                        output = self.model(**data)
                        loss = output.loss
                        logits = output.logits
                        probability = torch.softmax(logits, dim=-1)
                        prediction = torch.argmax(probability, dim=-1)
                        # self.f1_score.add_batch(predictions=prediction, references=data["labels"])
                        # self.preicision.add_batch(predictions=prediction, references=data["labels"])
                        # self.recall.add_batch(predictions=prediction, references=data["labels"])

                        metricDev = {
                            "f1Score": f1_score(np.array(prediction.detach().cpu().tolist()), np.array(data["labels"].detach().cpu().tolist()),average="macro"),
                            "recall": recall_score(np.array(prediction.detach().cpu().tolist()), np.array(data["labels"].detach().cpu().tolist())),
                            "precision": precision_score(np.array(prediction.detach().cpu().tolist()), np.array(data["labels"].detach().cpu().tolist()))
                        }
                        self.metrics_prc.append(metricDev)
        

        with open("metrics.json", "w") as fp:
            json.dump(self.metrics_prc, fp, indent=4)


        # save the huggingface model
        if not os.path.isdir(CONFIG.SAVE_MODEL_PATH):
            os.mkdir(CONFIG.SAVE_MODEL_PATH)

        self.model.save_pretrained(CONFIG.SAVE_MODEL_PATH)
                





training().trainModel()
