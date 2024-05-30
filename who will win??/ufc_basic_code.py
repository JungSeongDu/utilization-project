import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import he_normal
from keras.layers import Dropout

# UFC 데이터 불러오기
ufc_data = pd.read_csv("study/ufc_data.csv")

# 특성과 타겟 분리
X = ufc_data[['player1_height','player1_weight','player2_height','player2_weight']]
y = ufc_data['winner']  # 예측하려는 타겟 값

#print(X)
#print(y)

# 데이터 전처리: 특성 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#X_scaled = scale(X)

# 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2)

# 다중 레이어 퍼셉트론 모델 정의
# 모델 정의
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu', kernel_initializer=he_normal()))  # 첫 번째 은닉층
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_initializer=he_normal()))  # 두 번째 은닉층
model.add(Dropout(0.2))
#model.add(Dense(16, activation='relu', kernel_initializer=he_normal()))  # 세 번째 은닉층
#model.add(Dropout(0.2))
#model.add(Dense(8, activation='relu', kernel_initializer=he_normal()))  # 네 번째 은닉층
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer=he_normal()))  # 출력층 (이진 분류)


# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1, validation_data=(X_val, y_val))

print('\n')

score = model.evaluate(X_train,y_train)
print('training accuracy : %.2f%% \n' %(score[1]*100))

score = model.evaluate(X_test,y_test)
print('test accuracy : %.2f%% \n' %(score[1]*100))

print("\n------------------입력----------------------\n")

# 사용자로부터 입력 받기
player_1 = input("BLUE 선수의 이름을 입력하세요: ")
player1_height = float(input("BLUE 선수의 키를 입력하세요: "))
player1_weight = float(input("BLUE 선수의 몸무게를 입력하세요: "))

print("\n")

player_2 = input("RED 선수의 이름을 입력하세요: ")
player2_height = float(input("RED 선수의 키를 입력하세요: "))
player2_weight = float(input("RED 선수의 몸무게를 입력하세요: "))

print("\n")

# 입력값을 모델이 요구하는 형태로 변환
input_data = [[player1_height, player1_weight, player2_height, player2_weight]]
input_scaled = scaler.transform(input_data)

# 승부 예측
player1_prediction = model.predict(input_scaled)
player2_prediction = 1 - player1_prediction  # 두 번째 선수의 예측은 1에서 첫 번째 선수의 예측을 뺌

percent_player1 = round(float(player1_prediction[0][0]) * 100, 2)
percent_player2 = round(float(player2_prediction[0][0]) * 100, 2)

print("\n------------------결과----------------------\n")

print("%s 선수가 이길 확률: %.2f%%" %(player_1, percent_player1))
print("%s 선수가 이길 확률: %.2f%%" %(player_2, percent_player2))


if player1_prediction > player2_prediction:
    print("%s 선수가 이길 것으로 예측됩니다." %player_1)
else:
    print("%s 선수가 이길 것으로 예측됩니다." %player_2)

print("\n")