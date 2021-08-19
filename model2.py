import numpy as np
import pandas as pd
import math
import random as rn
#import matplotlib.pyplot as plt
import pickle


data = pd.read_csv('outbreak_detect.csv')


# importing the preprocessing module from scikit-learn

from sklearn import preprocessing
LE= preprocessing.LabelEncoder()

# Fitting it to our dataset

data.Outbreak = LE.fit_transform(data.Outbreak)
data.head()

data = data.drop('Positive',axis=1)
data = data.drop('pf',axis=1)
data = data.drop('Rainfall',axis=1)
#Training
x = data.iloc[:,0:3]
y = data.iloc[:,3:4]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(x,y)




inputt = [float(x) for x in "33 22 73.71".split(' ')]
final = [np.array(inputt)]
prediction=clf.predict_proba(final)
output = '{0:.{1}f}'.format(prediction[0][1], 2)
if output>=str(0.5):
  print(1)
else:
  print(0)


pickle.dump(clf,open('model/model.pkl','wb'))
model=pickle.load(open('model/model.pkl','rb'))
print("SUcess loaded")