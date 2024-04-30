import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
import numpy as np
from pandas.plotting import scatter_matrix

#print(os.listdir())

df = pd.read_csv("study/data.csv")

#print(df)

df = pd.DataFrame(df[["BPrev", "B_Age", "B_Height", "B_Weight",
                      "RPrev", "R_Age", "R_Height", "R_Weight", "winner"]])
#print(df)

#print(df.isnull().any())

#결측치 보간 - 평균
df['B_Age'] = df['B_Age'].fillna(df['B_Age'].mean())
df['B_Height'] = df['B_Height'].fillna(df['B_Height'].mean())
df['R_Age'] = df['R_Age'].fillna(df['R_Age'].mean())



df["B_reach"] = df["B_Height"] + 5
df["R_reach"] = df["R_Height"] + 5

df = df[df['winner'] != 'no contest']
df = df[df['winner'] != 'draw']

df['winner'] = df['winner'].replace('red',0)
df['winner'] = df['winner'].replace('blue',1)

#print(df.isnull().any())

#print(df)

#전체적인 데이터의 형태를 파악
"""
df.hist()
plt.tight_layout()
plt.show()
"""


#특징변수와 목표변수의 관계
plt.subplots(3,4,figsize=(10,10))

# 각 특징의 변수의 밀도 차트를 그린다
for idx, col in enumerate(["BPrev", "B_Age", "B_Height", "B_Weight", "B_reach",
                           "RPrev", "R_Age", "R_Height", "R_Weight", "R_reach"]):
    ax = plt.subplot(3, 4, idx + 1)

    # Outcome에 따른 밀도 차트 그리기
    sns.kdeplot(df.loc[df.winner == 0, col], color='red', label="red win")
    sns.kdeplot(df.loc[df.winner == 1, col], color='blue', label="blue win")
    
    ax.set_title(col)
    # 범례 추가
    plt.legend()

plt.subplot(3, 4, 11).set_visible(False)
plt.subplot(3, 4, 12).set_visible(False)
plt.tight_layout()
plt.show()


#df['B_Age'] = df['B_Age'].fillna(df['B_Age'].mean())
#df['B_Height'] = df['B_Height'].fillna(df['B_Height'].mean())
#df['R_Age'] = df['R_Age'].fillna(df['R_Age'].mean())




#상관관계 

corr_matrix = df.corr(numeric_only=True)

print(corr_matrix["winner"].sort_values(ascending=False))


B_attribute = ['B_Age','BPrev','B_Height','B_Weight','B_reach']
R_attribute = ['R_Age','RPrev','R_Height','R_Weight','R_reach']


# scatter_matrix 그리기
scatter_matrix(df[B_attribute], figsize=(12, 8))

scatter_matrix(df[R_attribute], figsize=(12, 8))
plt.show()
