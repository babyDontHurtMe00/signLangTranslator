import numpy as np
import cv2 #opencv
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline

#storing paths to all csv files
import os
csv_name_list=[]

for entry in os.scandir("./trainingData"):
    csv_name_list.append(entry.path)
# csv_name_list

#giving nos. to each letter/no.

class_dict = {}
count = 0
for i in range(10):
    class_dict[str(i)]=i
    
for i in range(65,91):
    class_dict[chr(i)] = i
# class_dict



import csvLibrary as cl
X=[]
y=[]

for file in csv_name_list:        
    path = file
    # print(path[15])
    csvData = cl.dread(path)

    for rowEntry in csvData:
        # print(rowEntry)
        tempRow = []
        for handPoint, coords in rowEntry.items():
            if coords != '':
                coords = list(coords[1:-1].split(","))
                # print(float(handPoint), int(coords[0]),int(coords[1]))
                tempRow.extend([float(handPoint), int(coords[0]),int(coords[1])])
            else:
                tempRow.extend([float(handPoint),-600,-600])
        X.append(tempRow)
        y.append(path[15])
# print(X)
# print(y,len(y))


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pandas as pd


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)

pipe = make_pipeline(StandardScaler(), svm.SVC(gamma='auto',probability=False))
pipe.fit(X_train, y_train)
print("SCORE:",pipe.score(X_test, y_test))

# print(classification_report(y_test, pipe.predict(X_test),zero_division=np.nan))


import joblib 
# Save the model as a pickle in a file 
joblib.dump(pipe, './artifacts/saved_model.pkl') 


import json
with open("./artifacts/class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))

