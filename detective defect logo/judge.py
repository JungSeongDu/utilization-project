import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# 모델 로드
model1 = keras.models.load_model("logo_detect1.keras")
model2 = keras.models.load_model("logo_detect2.keras")
model3 = keras.models.load_model("logo_detect3.keras")

INPUT_SIZE  = 256

# 실시간 데이터 전처리 함수
def preprocess_image(image):
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))  # 이미지 크기 조정
    image = image.astype("float32") / 255.0  # 정규화
    image = np.expand_dims(image, axis=-1)  # 채널 차원 추가
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# 실시간 예측 함수
def predict_realtime(image):
    preprocessed_image = preprocess_image(image)
    prediction1 = model1.predict(preprocessed_image)
    prediction2 = model2.predict(preprocessed_image)
    prediction3 = model3.predict(preprocessed_image)
    ensemble_prediction = (prediction1 + prediction2 + prediction3) / 3
    final_prediction = 1 if ensemble_prediction > 0.5 else 0  # 이진 분류 결과 계산
    return final_prediction

# 관심 영역 설정 (예: 화면 중앙의 세로로 길쭉한 사각형 영역)
def get_roi(frame):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = w // 4, h // 6, 3 * w // 4, 5 * h // 6  # 화면 중앙의 세로로 길쭉한 사각형 영역
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

# 웹캠을 통해 실시간으로 이미지 가져오기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi, (x1, y1, x2, y2) = get_roi(frame)
    prediction = predict_realtime(roi)
    label = "Normal" if prediction == 1 else "Defective"
    
    # 예측 결과에 따라 글씨 색상 설정
    color = (0, 255, 0) if label == "Normal" else (0, 0, 255)  # Normal: Green, Defective: Red

    # 예측 결과를 화면에 표시
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 관심 영역 표시
    cv2.imshow("Real-time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
