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

everitylist = le.fit_transform(list(severity["Symptom"]))
def take_input():
    symptom_1 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptom_2 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptom_3 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptom_4 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptoms = symptom_1 + " " + symptom_2 + " " + symptom_3 + " " + symptom_4
    symptoms_list = symptoms.split()
    symptoms_list = le.transform(symptoms_list)
    symptoms_list = np.array(list(zip(symptoms_list)))
    return symptoms_list, type(symptoms_list), Symptom_1

X = np.array(list(zip(Symptom_1, Symptom_2, Symptom_3, Symptom_4)))
y = np.array(dataset[predict])

#X = np.array(dataset.drop([predict], 1))
#y = np.array(dataset[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=25)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("The Accuracy of this prediction is", acc)
pred = model.predict(x_test)