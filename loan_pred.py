import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,MinMaxScaler


doc = pd.read_csv("C:/Users/STAFF/Desktop/Git_Repos/train_load_predicition.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(doc.head())
print(f"Statistical measures:\n {doc.describe()}")

doc = doc.dropna()
print(doc.head())
doc = doc.drop(columns="Loan_ID")

#cast the dtypes into a list of tuples
dtypes_tuple = [(column, dtype) for column, dtype in zip(doc.columns, doc.dtypes)]

categorical_columns = [i[0] for i in dtypes_tuple if i[1]=="object"]
print(f"\ncategorical columns : \n {categorical_columns}")

encoder = LabelEncoder()

# Applying LabelEncoder to the categorical columns
for j in categorical_columns:
    doc[j] = encoder.fit_transform(doc[j])

X = doc.drop(columns="Loan_Status") #features
y = doc["Loan_Status"] #labels

minmax = MinMaxScaler()
X = minmax.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=42)