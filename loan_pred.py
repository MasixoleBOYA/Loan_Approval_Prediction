import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score

doc = pd.read_csv("C:/Users/STAFF/Desktop/Git_Repos/train_load_predicition.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# print(doc.head())
print(f"Statistical measures:\n {doc.describe()}")

