import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.utils import resample
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import joblib


#데이터 불러오기
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


# 각 클래스의 샘플 수 계산
red_count = df[df['winner'] == 'red'].shape[0]
blue_count = df[df['winner'] == 'blue'].shape[0]

print("Original red count:", red_count)
print("Original blue count:", blue_count)

print("\n")

# 클래스 간의 샘플 수가 다른 경우 샘플링하여 클래스 간의 샘플 수를 맞춤
if red_count > blue_count:
    df_red_downsampled = resample(df[df['winner'] == 'red'], replace=False, n_samples=blue_count)
    df = pd.concat([df_red_downsampled, df[df['winner'] == 'blue']])
elif blue_count > red_count:
    df_blue_downsampled = resample(df[df['winner'] == 'blue'], replace=False, n_samples=red_count)
    df = pd.concat([df[df['winner'] == 'red'], df_blue_downsampled])
else:
    pass  # 클래스 간의 샘플 수가 같은 경우 그대로 사용


# 수정된 데이터프레임에서 클래스별로 샘플 수 출력
red_count_balanced = df[df['winner'] == 'red'].shape[0]
blue_count_balanced = df[df['winner'] == 'blue'].shape[0]


print("Balanced red count:", red_count_balanced)
print("Balanced blue count:", blue_count_balanced)


df['winner'] = df['winner'].replace('red',0)
df['winner'] = df['winner'].replace('blue',1)

print(df.describe())


#print(ufc_data['winner'])

# 특성과 타겟 분리

X = df[['B_Age','BPrev','B_Height','B_Weight','B_reach',
              'R_Age','RPrev','R_Height','R_Weight','R_reach']]

y = df['winner']  # 예측하려는 타겟 값

#print(X)
#print(y)

# 데이터 전처리: 특성 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_scaled = scale(X)

# 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=0)


y_train = y_train.astype(int)

#print("X_train shape:", X_train.shape)
#print("y_train shape:", y_train.shape)
#print(y_train)


# 다중 레이어 퍼셉트론 모델 정의
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=10))  # 첫 번째 은닉층
model.add(Dropout(0.2))  # 드롭아웃 추가 (20%의 뉴런을 랜덤하게 비활성화)
model.add(Dense(16, activation='relu'))  # 두 번째 은닉층
model.add(Dropout(0.2))  # 드롭아웃 추가
model.add(Dense(1, activation='sigmoid'))  # 출력층 (이진 분류)


# 모델 컴파일
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 요약 정보 출력
model.summary()



# 모델 학습
model.fit(X_train, y_train, epochs=50 , batch_size=32, verbose=1, validation_data=(X_val, y_val))

print('\n')

score = model.evaluate(X_train,y_train)
print('training accuracy : %.2f%% \n' %(score[1]*100))

score = model.evaluate(X_test,y_test)
print('test accuracy : %.2f%% \n' %(score[1]*100))



#혼동행렬
# 모델의 예측 확률
y_test_pred_probs = model.predict(X_test)

# 예측 확률 중에서 가장 높은 값을 갖는 클래스의 인덱스
y_test_pred = (y_test_pred_probs > 0.5).astype(int)

# 혼동 행렬을 계산
c_matrix = confusion_matrix(y_test, y_test_pred)

# 시각화
ax = sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['red win', 'blue win'], 
                 yticklabels=['red win', 'blue win'], 
                 cbar=False)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()

#정밀도
pre_score = precision_score(y_test, y_test_pred)
print("정밀도 : %.2f%%" %(pre_score*100))

#재현률
re_call_score = recall_score(y_test,y_test_pred)
print("재현률 : %.2f%%" %(re_call_score*100))

#f1 점수
f_1_score = f1_score(y_test, y_test_pred)
print("f1 점수 : %.2f%% \n" %(f_1_score*100))


joblib.dump(scaler, "scaler.pkl")
model.save("ufc_model.keras")


