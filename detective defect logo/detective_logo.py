import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from scipy.stats import mode
from matplotlib import pyplot as plt  
import random  

# 초매개변수 정의
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 256
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10


# ImageDataGenerator를 사용하여 이미지 로드 및 전처리
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

training_set = train_datagen.flow_from_directory('logo/Dataset/LogoImages/Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = 32,
                                                class_mode = 'binary',
                                                shuffle=False)

test_set = test_datagen.flow_from_directory('logo/Dataset/LogoImages/Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = 32,
                                             class_mode = 'binary',
                                             shuffle=False)


# 모델 아키텍처 정의
def build_model_1():
    model1 = keras.Sequential([
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    return model1


def build_model_2():
    model2 = keras.Sequential([
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),  # 추가된 은닉층
        layers.Dropout(0.2),  # 드롭아웃 층 추가
        layers.Dense(1, activation="sigmoid")
    ])
    return model2

# 모델 빌드
model1 = build_model_1()
model2 = build_model_2()

# 모델 컴파일
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 각 모델에 훈련
model1.fit(training_set, epochs=15 , verbose=1)
model2.fit(training_set, epochs=15, verbose=1)

# 모델 저장
model1.save("logo_detect1.keras")
model2.save("logo_detect2.keras")

# 앙상블을 위해 각 모델의 예측을 수행
#test_set.reset()
predictions1 = model1.predict(test_set)
predictions2 = model2.predict(test_set)

# 실제 라벨
test_labels = test_set.classes

# 클래스 레이블 확인
print("클래스 레이블:", training_set.class_indices)


# 이진 클래스 분류를 위한 예측 결과 처리
predictions1 = np.where(predictions1 > 0.5, 1, 0)
predictions2 = np.where(predictions2 > 0.5, 1, 0)

# 모델 1의 정확도 계산
accuracy_model1 = accuracy_score(test_labels, predictions1)
print(f"Model 1 Accuracy: {accuracy_model1:.2f}")

# 모델 2의 정확도 계산
accuracy_model2 = accuracy_score(test_labels, predictions2)
print(f"Model 2 Accuracy: {accuracy_model2:.2f}")

# 앙상블 예측 생성
ensemble_predictions = (predictions1 + predictions2) / 2
final_predictions = np.where(ensemble_predictions > 0.5, 1, 0)

# 앙상블 모델의 정확도 계산
accuracy_ensemble = accuracy_score(test_labels, final_predictions)
print(f"Ensemble Model Accuracy: {accuracy_ensemble:.2f}")

#----------------------------------------------------------------------------------------


def plot_on_grid(test_set, idx_to_plot, img_size=(INPUT_SIZE, INPUT_SIZE),title=""):  
    num_samples = min(4, len(idx_to_plot))  # 샘플 가능한 최대 수로 조정
    if num_samples > 0:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))  
        sampled_indices = random.sample(idx_to_plot, num_samples)
        for i, idx in enumerate(sampled_indices):  
            img = test_set.__getitem__(idx)[0].reshape(img_size[0], img_size[1], 3)   
            ax[int(i/2), i%2].imshow(img)
            ax[int(i/2), i%2].axis('off')
        fig.suptitle(title,fontsize=20)  # 제목 추가  
        plt.show()
    else:
        print("샘플링할 요소가 충분하지 않습니다.")


test_set = test_datagen.flow_from_directory('logo/Dataset/LogoImages/Test/',
                                             target_size=(INPUT_SIZE, INPUT_SIZE),
                                             batch_size=1,
                                             class_mode='binary',
                                             shuffle=False)

strongly_right_idx = []   
weakly_wrong_idx = []

for i in range(len(test_set)):  
    img = test_set.__getitem__(i)[0]  
    pred_prob = ensemble_predictions[i][0]  # 앙상블 예측 확률 가져오기
    pred_label = int(pred_prob > 0.5)  
    actual_label = int(test_set.__getitem__(i)[1][0])  
    
    if pred_label != actual_label and (pred_prob > 0.3 and pred_prob < 0.6): 
        weakly_wrong_idx.append(i)  
    elif pred_label == actual_label and (pred_prob > 0.7 or pred_prob < 0.1): 
        strongly_right_idx.append(i)  

    if (len(strongly_right_idx) >= 1 and len(weakly_wrong_idx) >= 1): 
        break  

# 디버깅 정보 출력
print(f"최종 strongly_right_idx 길이: {len(strongly_right_idx)}")

plot_on_grid(test_set, strongly_right_idx, title="Strongly Right Predictions")

plot_on_grid(test_set, weakly_wrong_idx,title="Weakly Wrong Predictions")





