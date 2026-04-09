# ruff: noqa: E402 # to ignore "imports not on top of the file" warning

import pandas as pd

# LOAD DATASET

df = pd.read_csv("titanic.csv")
print(df.head())



# find features that matter - which columns are useful - FEATURE ENGINEERING

print(df.groupby("Pclass")["Survived"].mean()) # gives the prob of that group surviving - cause survived is either 0 or 1

print(df.groupby("Sex")["Survived"].mean())

# find the no.of values that's null in a specific column
# decide what to do with them: drop column - drop row - fill in a reasonable value
print(df.isnull().sum())



# PREPARE DATA for training

# drop columns that won't help
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
# fill missing ages with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())
# fill missing embarked value with the most common
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Now another problem - checkout the sex and Embarked columns - they're not numbers!
# change them to numbers - ENCODING

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})



# check
print(df.head())
print(df.isnull().sum())



# SPLIT FEATURES
# X - features (what the model sees to make a prediction)
# Y - target (what the model is trying to predict)
# then split into TRAIN and TEST

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
Y = df["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} passengers")
print(f"Testing set: {len(X_test)} passengers")



# BUILD MODEL
# scikit-learn
# create model - train - predict

from sklearn.metrics import accuracy_score

# DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42) # create
dt.fit(X_train, Y_train) # train
Y_pred = dt.predict(X_test) # predict
print(f"Decision Tree Accuracy: {accuracy_score(Y_test, Y_pred):.4f}") # find accuracy - how good is it?

# K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

# SUPPORT VECTOR MACHINES
from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")



# OUTPUT

# Decision Tree Accuracy: 0.7765
# KNN Accuracy: 0.6927
# SVM Accuracy: 0.6536
# Random Forest Accuracy: 0.8156

# Random forest was the obvious winner
# but KNN and SVM did worse than simple Decision Tree - that seems wrong
# What happened ? Culprit - 'Fare' column
# in 'Fare' values range from 0 - 512, now 'Sex' it's 0 or 1.
# So when KNN or SVM measures distance between two passengers - the difference on 500 in 'Fare' drowns the difference of 1 in 'Sex'
# the model thinks - 'Fare' matters 500x more than 'Sex'. just because the numbers are big.
# Decision Tree don't care about scale. They just split on a certain threshold for each parameter. That's why the tree was fine.
# KNN and SVM are distance based model - sensitive to scale.
# FIX: FEATURE SCALING - squish all features to the same range.

# FEATURE SCALING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # learn stats + scale
X_test_scaled = scaler.transform(X_test) # just scale (using training stats) - cause in irl we won't have the test data, so we can't learn from it

# KNN - with scaled features
knn.fit(X_train_scaled, Y_train)
Y_pred = knn.predict(X_test_scaled)
print(f"KNN-Scaled Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

# SVM - with scaled features
svm.fit(X_train_scaled, Y_train)
Y_pred = svm.predict(X_test_scaled)
print(f"SVM-Scaled Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")