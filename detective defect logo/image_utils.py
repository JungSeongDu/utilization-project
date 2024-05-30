import os
import random
import shutil
import piexif


def train_test_split(src_folder, train_size = 0.8):
	# 기존 폴더를 삭제하고 깨끗한 상태로 시작한다. 
	shutil.rmtree(src_folder+'Train/Nomal/', ignore_errors=True)
	shutil.rmtree(src_folder+'Train/Defect/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Nomal/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Defect/', ignore_errors=True)

	# train 폴더와 test 폴더를 새로 생성한다.
	os.makedirs(src_folder+'Train/Nomal/')
	os.makedirs(src_folder+'Train/Defect/')
	os.makedirs(src_folder+'Test/Nomal/')
	os.makedirs(src_folder+'Test/Defect/')

	# 이미지 개수를 가져온다.
	_, _, nomal_images = next(os.walk(src_folder+'Nomal/'))
	num_nomal_images = len(nomal_images)
	num_nomal_images_train = int(train_size * num_nomal_images)
	num_nomal_images_test = num_nomal_images - num_nomal_images_train

	_, _, defect_images = next(os.walk(src_folder+'Defect/'))
	num_defect_images = len(defect_images)
	num_defect_images_train = int(train_size * num_defect_images)
	num_defect_images_test = num_defect_images - num_defect_images_train

	# 이미지를 무작위로 골라 train 폴더와 test 폴더로 나눈다.
	nomal_train_images = random.sample(nomal_images, num_nomal_images_train)
	for img in nomal_train_images:
		shutil.copy(src=src_folder+'Nomal/'+img, dst=src_folder+'Train/Nomal/')
	nomal_test_images  = [img for img in nomal_images if img not in nomal_train_images]
	for img in nomal_test_images:
		shutil.copy(src=src_folder+'Nomal/'+img, dst=src_folder+'Test/Nomal/')

	defecet_train_images = random.sample(defect_images, num_defect_images_train)
	for img in defecet_train_images:
		shutil.copy(src=src_folder+'Defect/'+img, dst=src_folder+'Train/Defect/')
	defect_test_images  = [img for img in defect_images if img not in defecet_train_images]
	for img in defect_test_images:
		shutil.copy(src=src_folder+'Defect/'+img, dst=src_folder+'Test/Defect/')

	# 데이터셋에서 잘못된 EXIF 데이터를 제거한다.
	#remove_exif_data(src_folder+'Train/')
	#remove_exif_data(src_folder+'Test/')
	
# 마이크로소프트 데이터셋에 있는 잘못된 EXIF 데이터를 제거하는 헬퍼 함수
def remove_exif_data(src_folder):
	_, _, nomal_images = next(os.walk(src_folder+'Nomal/'))
	for img in nomal_images:
		try:
			piexif.remove(src_folder+'Nomal/'+img)
		except:
			pass

	_, _, defect_images = next(os.walk(src_folder+'Defect/'))
	for img in defect_images:
		try:
			piexif.remove(src_folder+'Defect/'+img)
		except:
			pass



src_folder = 'logo/Dataset/LogoImages/'
train_test_split(src_folder)