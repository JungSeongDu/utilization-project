from image_utils import train_test_split
import matplotlib
matplotlib.use("TkAgg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import random
import os

#테스트 이미지 / 훈련 이미지 분리
"""
src_folder = 'logo/Dataset/LogoImages/'
train_test_split(src_folder)
"""



image_generator = ImageDataGenerator(rotation_range = 30,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')


fig, ax = plt.subplots(2,3, figsize=(20,10))
all_images = []

_, _, defect_images = next(os.walk('logo/Dataset/LogoImages/Train/Defect/'))
random_img = random.sample(defect_images, 1)[0]
print(random_img)
random_img = plt.imread('logo/Dataset/LogoImages/Train/Defect/'+random_img)
all_images.append(random_img)

random_img = random_img.reshape((1,) + random_img.shape)


sample_augmented_images = image_generator.flow(random_img)

for _ in range(5):
	augmented_imgs = sample_augmented_images.__next__()
	for img in augmented_imgs:
		all_images.append(img.astype('uint8'))

for idx, img in enumerate(all_images):
	ax[int(idx/3), idx%3].imshow(img)
	ax[int(idx/3), idx%3].axis('off')
	if idx == 0:
		ax[int(idx/3), idx%3].set_title('Original Image')
	else:
		ax[int(idx/3), idx%3].set_title('Augmented Image {}'.format(idx))


plt.show()