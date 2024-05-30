import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt

# 모든 열을 생략 없이 출력하도록 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#print(os.listdir())

df = pd.read_csv("study/ufc.csv")

#print(df.describe())

#df['winner']

#print(df['winner'])

#df['winner'] = df['winner'].replace('red',0)

#df['winner'] = df['winner'].replace('blue',1)

#print(df['winner'])

"""
df['B_Age'] = df['B_Age'].fillna(df['B_Age'].mean())
df['B_Height'] = df['B_Height'].fillna(df['B_Height'].mean())
df['R_Age'] = df['R_Age'].fillna(df['R_Age'].mean())
"""

#print(df.isnull().any())

"""
a = df.isnull().sum(axis=0)
print(a)
"""


#print(df.isnull().sum(axis=0))


"""
for col in df.columns:
    missing_row = df.loc[df[col] == 0].shape[0]
    print(col + ":" + str(missing_row))
"""

"""
draw_row = df.loc[df['winner'] == 'draw'].shape[0]
print("draw : " + str(draw_row) + "개")


no_contest_row = df.loc[df['winner'] == 'no contest'].shape[0]
print("no contest : " + str(no_contest_row) + "개")
"""


df = df[df['winner'] != 'no contest']
df = df[df['winner'] != 'draw']


df['B_Age'] = df['B_Age'].fillna(df['B_Age'].mean())
df['B_Height'] = df['B_Height'].fillna(df['B_Height'].mean())
df['R_Age'] = df['R_Age'].fillna(df['R_Age'].mean())



df['winner'] = df['winner'].replace('red',0)
df['winner'] = df['winner'].replace('blue',1)

corr_matrix = df.corr(numeric_only=True)

print(corr_matrix["winner"].sort_values(ascending=False))


B_attribute = ['B_Age','BPrev','B_Height','B_Weight','B_reach']
R_attribute = ['R_Age','RPrev','R_Height','R_Weight','R_reach']


# scatter_matrix 그리기
scatter_matrix(df[B_attribute], figsize=(12, 8))

scatter_matrix(df[R_attribute], figsize=(12, 8))
plt.show()