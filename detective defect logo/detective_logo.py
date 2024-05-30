import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 초매개변수 정의
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10

# 데이터 경로 설정
train_dir = "logo/Dataset/Logoimages/train"
test_dir = "logo/Dataset/Logoimages/test"

# ImageDataGenerator를 사용하여 이미지 로드 및 전처리
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

training_set = train_datagen.flow_from_directory('logo/Dataset/LogoImages/Train/',
                                                target_size = (28, 28),
                                                batch_size = 16,
                                                class_mode = 'binary',
                                                shuffle=False)

test_set = test_datagen.flow_from_directory('logo/Dataset/LogoImages/Test/',
                                             target_size = (28, 28),
                                             batch_size = 16,
                                             class_mode = 'binary',
                                             shuffle=False)


# 모델 아키텍처 정의
def build_model_1():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


def build_model_2():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 3)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# 모델 빌드
model1 = build_model_1()
model2 = build_model_2()

# 모델 컴파일
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 각 모델에 서로 다른 서브셋을 사용하여 훈련
model1.fit(training_set, epochs=15 , verbose=1)
model2.fit(training_set, epochs=15, verbose=1)

# 모델 저장
model1.save("logo_detect1.keras")
model2.save("logo_detect2.keras")

# 앙상블을 위해 각 모델의 예측을 수행
#test_set.reset()
predictions1 = model1.predict(test_set)
predictions2 = model2.predict(test_set)

# 클래스 레이블 확인
print("클래스 레이블:", training_set.class_indices)

# 각 모델의 예측을 평균하여 앙상블 예측 생성
ensemble_predictions = (predictions1 + predictions2) / 2

# 최종 예측 결과
final_predictions = np.where(ensemble_predictions > 0.5, 1, 0)

# 정확도 평가
accuracy = np.mean(final_predictions == test_set.classes)
print("앙상블 정확도:", accuracy)

