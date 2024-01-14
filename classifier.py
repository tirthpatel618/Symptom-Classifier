import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("dataset.csv")
dataset = dataset[["Disease", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]]
severity = pd.read_csv("Symptom-severity.csv")
severity = severity[["Symptom", "weight"]]
predict = "Disease"

le = preprocessing.LabelEncoder()
Symptom_1 = le.fit_transform(list(dataset["Symptom_1"]))

Symptom_2 = le.fit_transform(list(dataset["Symptom_2"]))

Symptom_3 = le.fit_transform(list(dataset["Symptom_3"]))

Symptom_4 = le.fit_transform(list(dataset["Symptom_4"]))
