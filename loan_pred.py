import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix

doc = pd.read_csv("C:/Users/STAFF/Desktop/Git_Repos/train_load_predicition.csv")

#Visualizing the data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(f"Data shape: {doc.shape}")
print(f"Data head:\n {doc.head()}")
print(f"\nStatistical measures:\n {doc.describe()}")

doc = doc.dropna()
doc = doc.drop(columns="Loan_ID")

print(f"\nData shape (After cleaning: {doc.shape}")
print(f"\nData head (After cleaing):\n {doc.head()}")
print(f"\nStatistical measures (After cleaning):\n {doc.describe()}")

#cast the dtypes into a list of tuples
dtypes_tuple = [(column, dtype) for column, dtype in zip(doc.columns, doc.dtypes)]

categorical_columns = [i[0] for i in dtypes_tuple if i[1]=="object"]
print(f"\ncategorical columns : \n {categorical_columns}")

encoder = LabelEncoder()

#Applying LabelEncoder to the categorical features
for j in categorical_columns:
    doc[j] = encoder.fit_transform(doc[j])

# Correlation heatmap matrix
plt.figure(figsize=(10, 8))
sns.heatmap(doc.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
# plt.show()

#Splitting the data
X = doc.drop(columns="Loan_Status") #features
y = doc["Loan_Status"] #labels

minmax = MinMaxScaler()
X = minmax.fit_transform(X)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size= 0.60, random_state=None)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

#Model training
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
model_output = classifier.predict(X_test)

#Evaluation metrics
train_score = classifier.score(X_train, y_train)
test_score = classifier.score(X_test, y_test)
precision = precision_score(y_test, model_output)
sensitivity = recall_score(y_test, model_output)
f1 = f1_score(y_test, model_output)
conf_matrix = confusion_matrix(y_test,model_output)

print(f"training score: {round(train_score*100,3)}%")
print(f"test score: {round(test_score*100,3)}%")
print(f"\nprecision score: {round(precision,3)}")
print(f"recall score: {round(sensitivity,3)}")
print(f"f1 score: {round(f1,3)}")
print(f"\nconfusin matrix : \n {conf_matrix}")

# Plot the confusion matrix using seaborn
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()