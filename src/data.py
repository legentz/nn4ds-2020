from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os, shutil
from sys import exit
import numpy as np

'''
[...]
'''
class EMData():
	def __init__(self, data_path):

		# check whether path does exists
		assert os.path.exists(data_path)

		self.data_path = data_path
		self.dataset_to_folders = {
			'train-volume.tif': 'train/img',
			'train-labels.tif': 'labels/img',
			'test-volume.tif': 'test/img'
		}
	
	def __set_folders_up(self, overwrite=False):

		# prepare folders (labels, train, test)
		for f in self.dataset_to_folders.values():
			f_path = os.path.join(self.data_path, f)
			
			# delete folders if not empty and if overwrite is True
			if os.path.exists(f_path):

				# we care only about not empty folders
				if len(os.listdir(f_path)) > 0:

					# cannot overwrite folders (and their content)
					# if overwrite not requested by the user
					if not overwrite: raise Exception('One or all folders are not empty. Set option overwrite=True to solve this')
					
					# overwrite 
					shutil.rmtree(f_path, ignore_errors=True)
					os.makedirs(f_path)

			# folders do not exist. Need to create them
			else: os.makedirs(f_path)

	'''
	Unpack the original .tif multi-frame [...]
	'''
	def unpack(self, overwrite=False, save_as='.tif'):

		# Set folders structure up
		self.__set_folders_up(overwrite=overwrite)

		# need to be sure that we have each part of the dataset 
		for name, folder in self.dataset_to_folders.items():
			name_path = os.path.join(self.data_path, name)
			folder_path = os.path.join(self.data_path, folder)

			# need to be sure that both .tif and output folder do exist
			# TODO: move this part outside
			# TODO: os.path.join should be computed once
			assert os.path.exists(folder_path), folder_path + ' does not exists'
			assert os.path.exists(name_path), name_path + ' does not exists'

			# open .tif stack
			tif_stack = Image.open(name_path)
			tif_stack.load()

			print("Extracting no. {} frames from {}".format(tif_stack.n_frames, name))

			# save each piece in the stack
			for frame in range(tif_stack.n_frames):
				tif_stack.seek(frame)

				# ex. 11[.ext]
				frame_name = str(frame) + (save_as if save_as.startswith('.') else '.' + save_as)
				tif_stack.save(os.path.join(folder_path, frame_name))

	def load_training_data(self, data_augmentation=dict(), batch_size=3, image_size=(512, 512), seed=None):
		IDG = ImageDataGenerator(**data_augmentation)
		train_X = IDG.flow_from_directory(
			os.path.join(self.data_path, 'train'),
			class_mode=None,
			target_size=image_size,
			color_mode="grayscale",
			batch_size=batch_size,
			#save_to_dir=os.path.join(self.data_path, 'aug'),
			#save_prefix='train',
			seed=seed
		)
		train_y = IDG.flow_from_directory(
			os.path.join(self.data_path, 'labels'),
			class_mode=None,
			target_size=image_size,
			color_mode="grayscale",
			batch_size=batch_size,
			#save_to_dir=os.path.join(self.data_path, 'aug'),
			#save_prefix='label',
			seed=seed
		)

		# yeild data in batches
		for X, y in zip(train_X, train_y):
			yield X, y