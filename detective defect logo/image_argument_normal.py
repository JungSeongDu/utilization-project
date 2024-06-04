from image_utils import train_test_split
import matplotlib
matplotlib.use("TkAgg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import os

# 이미지 증강을 위한 ImageDataGenerator 생성
image_generator = ImageDataGenerator(rotation_range=30,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

# 증강할 이미지가 있는 디렉토리와 저장할 디렉토리 설정
src_dir = 'logo/Dataset/LogoImages/Normal/'
save_dir = 'logo/Dataset/LogoImages/Normal/'

# 디렉토리가 존재하지 않으면 생성
os.makedirs(save_dir, exist_ok=True)

# 이미지 증강 및 저장
for img_name in os.listdir(src_dir):
    # 이미지 파일 읽기
    img_path = os.path.join(src_dir, img_name)
    img = plt.imread(img_path)

    # 이미지 형태 변경
    img = img.reshape((1,) + img.shape)

    # 증강 이미지 생성을 위한 flow 생성
    sample_augmented_images = image_generator.flow(img, save_to_dir=save_dir, save_prefix='aug', save_format='jpg')

    # 증강 이미지 5개 생성 및 저장
    for _ in range(5):
        sample_augmented_images.__next__()

# 증강된 이미지 몇 개를 표시하기 위한 코드
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
all_images = []

# 증강된 이미지 리스트 가져오기
augmented_images = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith('aug')]

# 증강된 이미지 몇 개를 읽고 표시
for i in range(min(6, len(augmented_images))):
    img = plt.imread(augmented_images[i])
    all_images.append(img)

for idx, img in enumerate(all_images):
    ax[int(idx/3), idx%3].imshow(img)
    ax[int(idx/3), idx%3].axis('off')
    if idx == 0:
        ax[int(idx/3), idx%3].set_title('Sample Augmented Image')
    else:
        ax[int(idx/3), idx%3].set_title('Augmented Image {}'.format(idx))

plt.show()
