import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# print the column names
original_headers = list(nba.columns.values)
print(original_headers)

#print the first three rows.
print(nba[0:3])
nba = nba[(nba['G'] > 1) & (nba['MP'] > 1) & (nba['GS'] > 1)]
# "Position (pos)" is the class attribute we are predicting.
class_column = 'Pos'

#The dataset contains attributes such as player name and team name.
#We know that they are not useful for classification and thus do not
#include them as features.
feature_columns = ['FT', 'FG%', '3P', '3PA', \
    '3P%', '2P%','2P', 'eFG%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK']

#Pandas DataFrame allows you to select columns.
#We use column selection to split the data into features and class.
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

print(nba_feature[0:3])
print(list(nba_class[0:3]))

train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []
linearsvm = LinearSVC().fit(train_feature, train_class)
print("Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))
prediction = linearsvm.predict(test_feature)

print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))


scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))