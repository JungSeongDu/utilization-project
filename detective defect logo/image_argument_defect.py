import matplotlib
matplotlib.use("TkAgg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import os

# 이미지 증강을 위한 ImageDataGenerator 생성
image_generator = ImageDataGenerator(rotation_range=20,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     horizontal_flip=False,
                                     fill_mode='nearest')

"""
rotation_range=30:
이미지를 무작위로 회전할 각도 범위를 설정

width_shift_range=0.2:
이미지를 수평 방향으로 무작위로 이동시킬 범위를 설정

height_shift_range=0.2:
이미지를 수직 방향으로 무작위로 이동시킬 범위를 설정

zoom_range=0.2:
이미지를 무작위로 확대 또는 축소할 범위를 설정

horizontal_flip=False:
이미지를 무작위로 수평 반전

fill_mode='nearest':
이미지를 변형할 때 생기는 빈 공간을 채우는 방법을 설정
"""

# 증강할 이미지가 있는 디렉토리와 저장할 디렉토리 설정
src_dir = 'logo/Dataset/LogoImages/Defect/'
save_dir = 'logo/Dataset/LogoImages/Defect/'

# 디렉토리가 존재하지 않으면 생성
os.makedirs(save_dir, exist_ok=True)

# 이미지 증강 및 저장
for img_name in os.listdir(src_dir):
    # 이미지 파일 읽기
    img_path = os.path.join(src_dir, img_name)
    img = plt.imread(img_path)

    # 이미지 형태 변경
    img = img.reshape((1,) + img.shape)

    # 원본 파일 이름에서 확장자를 제거하여 접두사로 사용
    base_name = os.path.splitext(img_name)[0]

    # 증강 이미지 생성을 위한 flow 생성
    sample_augmented_images = image_generator.flow(
        img,
        batch_size=1,
        save_to_dir=None,  # 직접 저장
        save_prefix=base_name + '_aug',
        save_format='jpeg'
    )

    # 증강 이미지 생성 및 저장
    for i in range(10):
        augmented_img = sample_augmented_images.__next__()[0]
        augmented_img = array_to_img(augmented_img)

        # RGBA 모드인 경우 RGB 모드로 변환
        if augmented_img.mode == 'RGBA':
            augmented_img = augmented_img.convert('RGB')

        augmented_img.save(os.path.join(save_dir, f'{base_name}_aug_{i}.jpeg'))

# 증강된 이미지 몇 개를 표시하기 위한 코드
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
all_images = []

# 원본 이미지 추가
original_image_path = os.path.join(src_dir, os.listdir(src_dir)[0])
original_image = plt.imread(original_image_path)
all_images.append(('Original Image', original_image))

# 증강된 이미지 리스트 가져오기
augmented_images = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if '_aug' in f]

# 증강된 이미지 몇 개를 읽고 리스트에 추가
for i in range(min(10, len(augmented_images))):
    img = plt.imread(augmented_images[i])
    all_images.append((f'Augmented Image {i+1}', img))

# 동적으로 서브플롯 크기 결정
rows = int(np.ceil(len(all_images) / 3))
fig, ax = plt.subplots(rows, 3, figsize=(20, 10))

# 이미지 플롯
for idx, (title, img) in enumerate(all_images):
    row, col = divmod(idx, 3)
    ax[row, col].imshow(img)
    ax[row, col].axis('off')
    ax[row, col].set_title(title)

# 남은 서브플롯 숨기기
for i in range(len(all_images), rows * 3):
    fig.delaxes(ax.flatten()[i])

plt.show()
