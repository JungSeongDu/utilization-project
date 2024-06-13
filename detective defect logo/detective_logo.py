import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from scipy.stats import mode
from matplotlib import pyplot as plt  
import random  
from tensorflow.keras.applications import VGG16 # type: ignore
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# 초매개변수 정의
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 256
MAXPOOL_SIZE = 2
BATCH_SIZE = 32
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 15


# ImageDataGenerator를 사용하여 이미지 로드 및 전처리
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

training_set = train_datagen.flow_from_directory('logo/Dataset/LogoImages/Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = 32,
                                                class_mode = 'binary',
                                                shuffle=True)

test_set = test_datagen.flow_from_directory('logo/Dataset/LogoImages/Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = 32,
                                             class_mode = 'binary',
                                             shuffle=True)
#--------------------------------------------------------------------------------------------

# 이미지 로드 및 전처리를 위한 ImageDataGenerator 설정
datagen = ImageDataGenerator(rescale=1.0/255.0)

# 전체 데이터셋 로드
all_data = datagen.flow_from_directory('logo/Dataset/LogoImages/Train/',
                                       target_size=(INPUT_SIZE, INPUT_SIZE),
                                       batch_size=32,
                                       class_mode='binary',
                                       shuffle=True)

# 전체 데이터셋에서 파일 경로와 라벨 추출
file_paths = all_data.filepaths
labels = all_data.classes

# 데이터셋을 3개의 서브셋으로 나누기
total_size = len(file_paths)
indices = np.arange(total_size)
np.random.shuffle(indices)
split1 = int(total_size / 3)
split2 = int(total_size * 2 / 3)

train1_indices = indices[:split1]
train2_indices = indices[split1:split2]
train3_indices = indices[split2:]

train1_files = [file_paths[i] for i in train1_indices]
train1_labels = [labels[i] for i in train1_indices]

train2_files = [file_paths[i] for i in train2_indices]
train2_labels = [labels[i] for i in train2_indices]

train3_files = [file_paths[i] for i in train3_indices]
train3_labels = [labels[i] for i in train3_indices]

# 파일 경로와 라벨을 DataFrame으로 변환
def create_dataframe(file_paths, labels):
    df = pd.DataFrame({'filename': file_paths, 'class': labels})
    return df

train1_df = create_dataframe(train1_files, train1_labels)
train2_df = create_dataframe(train2_files, train2_labels)
train3_df = create_dataframe(train3_files, train3_labels)

# flow_from_dataframe을 사용하여 데이터 로드
def create_flow_from_dataframe(datagen, dataframe, batch_size=32):
    dataframe['class'] = dataframe['class'].astype(str)  # 정수 라벨을 문자열로 변환
    flow = datagen.flow_from_dataframe(dataframe,
                                       x_col='filename',
                                       y_col='class',
                                       target_size=(INPUT_SIZE, INPUT_SIZE),
                                       batch_size=batch_size,
                                       class_mode='binary',
                                       shuffle=True)
    return flow

train1_set = create_flow_from_dataframe(datagen, train1_df)
train2_set = create_flow_from_dataframe(datagen, train2_df)
train3_set = create_flow_from_dataframe(datagen, train3_df)

print("Train1 Set Size:", len(train1_files))
print("Train2 Set Size:", len(train2_files))
print("Train3 Set Size:", len(train3_files))

#---------------------------------------------------------------------------------------------



# 모델 아키텍처 정의
def build_model_1():
    model1 = keras.Sequential([
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),  # 추가된 은닉층
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    return model1


def build_model_2():
    model2 = keras.Sequential([
        layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),  # 추가된 은닉층
        layers.Dropout(0.3),  # 드롭아웃 층 추가
        layers.Dense(1, activation="sigmoid")
    ])
    return model2


def build_model_3():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

    # VGG16 기본 모델의 상위 레이어를 고정
    for layer in base_model.layers:
        layer.trainable = False

    # 새로운 Fully Connected 네트워크 추가
    model3 = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Global Average Pooling 층 추가
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    return model3

# 모델 빌드
model1 = build_model_1()
model2 = build_model_2()
model3 = build_model_3()

# 모델 컴파일
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model3.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 각 모델에 훈련
model1.fit(train1_set, epochs=EPOCHS, verbose=1)
model2.fit(train2_set, epochs=EPOCHS, verbose=1)
model3.fit(train3_set, epochs=EPOCHS, verbose=1)

# 모델 저장
model1.save("logo_detect1.keras")
model2.save("logo_detect2.keras")
model3.save("logo_detect3.keras")

# 앙상블을 위해 각 모델의 예측을 수행
#test_set.reset()
predictions1 = model1.predict(test_set)
predictions2 = model2.predict(test_set)
predictions3 = model3.predict(test_set)

# 실제 라벨
test_labels = test_set.classes

# 클래스 레이블 확인
print("클래스 레이블:", training_set.class_indices)


# 이진 클래스 분류를 위한 예측 결과 처리
predictions1 = np.where(predictions1 > 0.5, 1, 0)
predictions2 = np.where(predictions2 > 0.5, 1, 0)
predictions3 = np.where(predictions3 > 0.5, 1, 0)

# 모델 1의 정확도 계산
accuracy_model1 = accuracy_score(test_labels, predictions1)
print(f"Model 1 Accuracy: {accuracy_model1:.2f}")

# 모델 2의 정확도 계산
accuracy_model2 = accuracy_score(test_labels, predictions2)
print(f"Model 2 Accuracy: {accuracy_model2:.2f}")

# 모델 3의 정확도 계산
accuracy_model3 = accuracy_score(test_labels, predictions3)
print(f"Model 3 Accuracy: {accuracy_model2:.2f}")

# 앙상블 예측 생성
ensemble_predictions = (predictions1 + predictions2 + predictions3) / 3
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
                                             shuffle=True)

strongly_right_idx = []   
weakly_wrong_idx = []

for i in range(len(test_set)):  
    img = test_set.__getitem__(i)[0]  
    pred_prob = ensemble_predictions[i][0]  # 앙상블 예측 확률 가져오기
    pred_label = int(pred_prob > 0.5)  
    actual_label = int(test_set.__getitem__(i)[1][0])  
    
    if pred_label != actual_label and (pred_prob > 0.3 and pred_prob < 0.6): 
        weakly_wrong_idx.append(i)  
    elif pred_label == actual_label and (pred_prob >= 0.7 or pred_prob <= 0.3): 
        strongly_right_idx.append(i)  

    if (len(strongly_right_idx) >= 1 and len(weakly_wrong_idx) >= 1): 
        break  

# 디버깅 정보 출력
print(f"최종 strongly_right_idx 길이: {len(strongly_right_idx)}")

plot_on_grid(test_set, strongly_right_idx, title="Strongly Right Predictions")

plot_on_grid(test_set, weakly_wrong_idx,title="Weakly Wrong Predictions")





