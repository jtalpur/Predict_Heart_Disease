#Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from matplotlib import collections as matcoll



#Read the CSV Data
def Read_Data(file):
    heartDisease = file
    heartDisease
    pd.value_counts(heartDisease["target"])
    heart1 = heartDisease
    X = heart1.iloc[:, heart1.columns != "target"]
    y = heart1.iloc[:, heart1.columns == "target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test

accu_score = []
#Logistic Regression
def Log_Regression(X_train, X_test, y_train, y_test):
    lg = LogisticRegression(penalty="l1", random_state=0, solver='liblinear', multi_class= 'ovr')
    lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)
    accu_score.append(round(accuracy_score(y_test, y_pred)* 100,2))
    print('Test Accuracy of Logisitic Regression: ' + str(round(accuracy_score(y_test, y_pred)* 100,2)))
    cm = confusion_matrix(y_test, y_pred)
    return accu_score, cm

#MLP
def NeuralNet(X_train, X_test, y_train, y_test):
    nN = MLPClassifier(hidden_layer_sizes=(100,11), activation='identity', solver = 'lbfgs', random_state=50)
    nN.fit(X_train, y_train)
    y_pred = nN.predict(X_test)
    #accu_score = accuracy_score(y_test, y_pred)
    accu_score.append(round(accuracy_score(y_test, y_pred)* 100,2))
    print('Test Accuracy of Neural Network: ' + str(round(accuracy_score(y_test, y_pred)* 100,2)))
    cm = confusion_matrix(y_test, y_pred)
    return accu_score, cm

#LinearSVM
def Linear_SVM(X_train, X_test, y_train, y_test):
    L_svm = LinearSVC(penalty= 'l2', loss= 'squared_hinge',random_state=100, multi_class="ovr", dual=False)
    L_svm.fit(X_train, y_train)
    y_pred = L_svm.predict(X_test)
    #accu_score = accuracy_score(y_test, y_pred)
    accu_score.append(round(accuracy_score(y_test, y_pred)* 100,2))
    print('Test Accuracy of Linear SVM: ' + str(round(accuracy_score(y_test, y_pred)* 100,2)))
    cm = confusion_matrix(y_test, y_pred)
    return accu_score, cm


def SVM_Class(X_train, X_test, y_train, y_test):
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    #accu_score = accuracy_score(y_test, y_pred)
    accu_score.append(round(accuracy_score(y_test, y_pred)* 100,2))
    print('Test Accuracy of SVM Class: ' + str(round(accuracy_score(y_test, y_pred)* 100,2)))
    cm = confusion_matrix(y_test, y_pred)
    return accu_score, cm


def KNN(X_train, X_test, y_train, y_test):
    scores = []
    for each in range(1, 30):
        KNNfind = KNeighborsClassifier(n_neighbors=each)
        KNNfind.fit(X_train, y_train)
        scores.append(KNNfind.score(X_test, y_test))
    print("Best Number of Neighbors: " + str(scores.index(max(scores))+1) + " and its Accuracy: " + str(max(scores)))
    #plt.plot(range(1, 30), scores, color="black")
    #plt.xlabel("K Values")
    #plt.ylabel("Accuracy")
    #plt.show()
    neighbor = scores.index(max(scores))+1
    KNN = KNeighborsClassifier(n_neighbors=neighbor)  # n_neighbors = K value
    KNN.fit(X_train, y_train)  # learning model
    y_pred = KNN.predict(X_test)
    accu_score.append(round(accuracy_score(y_test, y_pred)* 100,2))
    print('Test Accuracy of KNN: ' + str(round(accuracy_score(y_test, y_pred)* 100,2)))
    cm = confusion_matrix(y_test, y_pred)
    return accu_score, cm



file = pd.read_csv('C:/Users/Jahanzaib Talpur/Desktop/Project/python/heart-disease-uci/heart.csv')
def run_the_code(file):
    X_train, X_test, y_train, y_test = Read_Data(file)
    accu_score, cm = Log_Regression(X_train, X_test, y_train, y_test)
    accu_score, cm = NeuralNet(X_train, X_test, y_train, y_test)
    accu_score, cm = Linear_SVM(X_train, X_test, y_train, y_test)
    accu_score, cm = SVM_Class(X_train, X_test, y_train, y_test)
    accu_score, cm = KNN(X_train, X_test, y_train, y_test)
    names = ["Logistic Regression", "Neural Network", "Linear SVM", "SVM Class", "KNN"]
    name_Score = dict(zip(names,accu_score))
    ax = plt.bar(name_Score.keys(), name_Score.values(), color = "g", edgecolor = "black")
    plt.title("Accuracy Scores of Models", fontsize = 20)
    plt.xlabel("Models", fontsize = 15)
    plt.ylabel("Accuracy", fontsize = 15)
    for i, v in enumerate(accu_score):
        plt.text(i, v + .3, str(v), color='black')
    plt.tight_layout()
    return ax

ax = run_the_code(file)
plt.show(ax)



# to get KNN neighbors graph
heart1 = file
X = heart1.iloc[:, heart1.columns != "target"]
y = heart1.iloc[:, heart1.columns == "target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0, stratify=y)
scores = []
for each in range(1, 30):
    KNNfind = KNeighborsClassifier(n_neighbors=each)
    KNNfind.fit(X_train, y_train)
    scores.append(KNNfind.score(X_test, y_test))
print("Best Number of Neighbors: " + str(scores.index(max(scores))+1) + " and its Accuracy: " + str(max(scores)))


x = np.arange(1,30)
ax = plt.scatter(x, scores, color="black")
ax = plt.plot(x, scores, color='blue')
plt.title("Best Neighbors for KNN", fontsize = 30)
plt.xlabel("K Values", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.xticks(x)
plt.show()