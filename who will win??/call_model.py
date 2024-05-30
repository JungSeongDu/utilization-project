from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


scaler = joblib.load("scaler.pkl")


# 모델 로드
loaded_model = load_model("ufc_model.keras")

def predict_result():
    print("\n")

    # 사용자로부터 입력 받기
    B_name = input("BLUE 선수의 이름을 입력하세요: ")
    B_Age = input("BLUE 선수의 나이를 입력하세요: ")
    BPrev = input("BLUE 선수의 경기횟수를 입력하세요: ")
    B_Height = float(input("BLUE 선수의 키를 입력하세요: "))
    B_Weight = float(input("BLUE 선수의 몸무게를 입력하세요: "))
    B_reach = B_Height + 5

    print("\n")

    R_name = input("RED 선수의 이름을 입력하세요: ")
    R_Age = input("RED 선수의 나이를 입력하세요: ")
    RPrev = input("RED 선수의 경기횟수를 입력하세요: ")
    R_Height = float(input("RED 선수의 키를 입력하세요: "))
    R_Weight = float(input("RED 선수의 몸무게를 입력하세요: "))
    R_reach = R_Height + 5

    print("\n")

    # 입력값을 DataFrame으로 변환
    input_data = pd.DataFrame({
        'B_Age': [B_Age],
        'BPrev': [BPrev],
        'B_Height': [B_Height],
        'B_Weight': [B_Weight],
        'B_reach': [B_reach],
        'R_Age': [R_Age],
        'RPrev': [RPrev],
        'R_Height': [R_Height],
        'R_Weight': [R_Weight],
        'R_reach': [R_reach]
    })
    
    # 데이터를 스케일링합니다.
    input_scaled = scaler.transform(input_data)

    # 승부 예측
    predictions = loaded_model.predict(input_scaled)

    B_prediction = predictions[0][0]  # 첫 번째 선수가 이길 확률
    R_prediction = 1 - B_prediction  # 두 번째 선수가 이길 확률

    percent_B = round(float(B_prediction) * 100, 2)
    percent_R = round(float(R_prediction) * 100, 2)

    print("\n------------------결과----------------------\n")

    print("%s 선수가 이길 확률: %.2f%%" % (B_name, percent_B))
    print("%s 선수가 이길 확률: %.2f%%" % (R_name, percent_R))

    print("\n")

    if percent_B > percent_R:
        print("%s 선수가 이길 것으로 예측됩니다." % B_name)
    else:
        print("%s 선수가 이길 것으로 예측됩니다." % R_name)

    print("\n")

# 사용자가 'exit'을 입력할 때까지 계속해서 예측
print("\033[1;31mWelcome!!!!\033[0m") 
print("\033[1;31mWho will win???\033[0m") 
while True:
    input_choice = input("\033[1;34mEnter 'predict' to predict result or 'exit' to quit: \033[0m")
    if input_choice.lower() == 'exit':
        print("Exiting...")
        print("Thank you for join our AI \n")
        break
    elif input_choice.lower() == 'predict':
        predict_result()
    else:
        print("Invalid input. Please enter 'predict' or 'exit'.")
