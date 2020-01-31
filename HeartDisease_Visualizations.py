#Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns




#Visualizations
file = pd.read_csv('C:/Users/Jahanzaib Talpur/Desktop/Project/python/heart-disease-uci/heart.csv')
heartDisease = file
Viz_HeartDisease = file
Viz_HeartDisease['target'] = Viz_HeartDisease['target'].replace([0,1], ["No","Yes"])
Viz_HeartDisease["gender"] = Viz_HeartDisease["sex"].replace([0,1], ["Female","Male"])
pd.value_counts(Viz_HeartDisease["target"])

#plot the heatmap using seaborn
ait_corr = Viz_HeartDisease.iloc[:, 1:12].corr(method = 'pearson')
sns.heatmap(ait_corr, xticklabels= ait_corr.columns.values, yticklabels= ait_corr.columns.values).set_title('Correlation Plot')

#Create subplot
#f, axes = plt.subplots(2,2)
rcParams['patch.force_edgecolor'] = True
#plot the count plot /bar plot using seaborn
axes0 = sns.countplot(x="target", data = Viz_HeartDisease)
axes0.set_xlabel("Target", fontsize = 20)
axes0.set_ylabel("Count", fontsize = 20)
axes0.set_title('Patients Count with and without Disease', fontsize = 30)
axes0.annotate(138, xy = (1, 138 + 3))
axes0.annotate(165, xy = (0, 165 + 3))
plt.tight_layout()


#plot gender bar plot
pd.value_counts(Viz_HeartDisease["gender"])
axes1 = sns.countplot(x="gender", data=Viz_HeartDisease)
axes1.set_xlabel("Gender", fontsize = 20)
axes1.set_ylabel("Count", fontsize = 20)
axes1.set_title("Patients Gender Count", fontsize = 30)
axes1.annotate(96, xy = (1, 96 + 3))
axes1.annotate(207, xy = (0, 207 + 3))
plt.tight_layout()


#plot the target by gender
axes2 = sns.countplot(x="target", hue = 'gender', data = Viz_HeartDisease)
axes2.set_xlabel('Target', fontsize = 20)
axes2.set_ylabel("Count", fontsize = 20)
axes2.set_title('Patients by Gender', fontsize = 30)
plt.tight_layout()


#plot age distribution
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'g'
axes3 = sns.distplot(heartDisease['age'], color = "#2ecc71")
axes3.set_xlabel('Age', fontsize = 20)
axes3.set_ylabel('Density', fontsize = 20)
axes3.set_title('Age Distribution Plot', fontsize = 30)
plt.tight_layout()

#Some calculations
min(heartDisease['age'])
max(heartDisease['age'])
ageDifference = max(heartDisease['age']) - min(heartDisease['age'])
np.mean(Viz_HeartDisease['age'])
np.median(Viz_HeartDisease['age'])

#binned the age distribution
binned = np.histogram(Viz_HeartDisease['age'], bins=4, range=None, normed=False, weights=None)
axes4 = sns.barplot(["29-40", "41-53","54-65", "66-77"], binned[0])
axes4.set_xlabel('Age Groups', fontsize = 20)
axes4.set_ylabel('Count', fontsize = 20)
axes4.set_title('Age Distribution Plot', fontsize = 30)
plt.tight_layout()

##
Yes = pd.DataFrame.reset_index(file.iloc[:164,:], drop=True)
No = pd.DataFrame.reset_index(file.iloc[165:,:], drop=True)

#f, axes = plt.subplots(1,2)
axes0 = sns.distplot(Yes['age'], color ="#e74c3c")
axes0.set_xlabel('Age', fontsize = 20)
axes0.set_ylabel('Density', fontsize = 20)
axes0.set_title('Age Distribution Plot of Patients with Disease', fontsize = 30)
axes3 = sns.distplot(No['age'], color = "#34495e")
axes3.set_xlabel('Age', fontsize = 20)
axes3.set_ylabel('Density', fontsize = 20)
axes3.set_title('Age Distribution Plot For Patients without Disease', fontsize = 30)
axes0.legend(labels= ['Yes', 'No'], fontsize = 15)
plt.tight_layout()