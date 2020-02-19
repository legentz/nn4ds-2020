from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os, shutil
from sys import exit
import numpy as np

'''
[...]
'''
class EMData():
	#def __init__(self, data_path, seed=None):
	def __init__(self, data_path):

		# check whether path does exists
		assert os.path.exists(data_path)

		# Same seed for reproducibility (no. matricula)
		#self.seed = seed

		# dataset 
		self.data_path = data_path
		self.dataset_to_folders = {
			'train-volume.tif': 'train/img',
			'train-labels.tif': 'labels/img',
			'test-volume.tif': 'test/img'
		}

		# data iterators (generators)
		self.data_generator = None
		self.split_train_for_validation = False
	
	# makes sure that folders structure is set up correctly
	# and it does not overwrite anything, unintentionally
	def _set_folders_up(self, overwrite=False):

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

	# it converts a Numpy array composed of real values [0, 1]
	# into a binary one using a threshold
	def _to_binary(self, data, threshold=0.6):
		assert type(data) == np.ndarray

		# check if it's already binary
		if ((data == 0) | (data == 1)).all():
			return data

		# transform data in binary using a threshold (binary classification )
		data[data < threshold] = 0 
		data[data >= threshold] = 1
		return data

	'''
	Unpack the original multi-frame .tif file
	'''
	def unpack(self, overwrite=False, save_as='.tif'):

		# Set folders structure up
		self._set_folders_up(overwrite=overwrite)

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

			# save each piece of the stack
			for frame in range(tif_stack.n_frames):
				tif_stack.seek(frame)

				# ex. 11[.ext]
				frame_name = str(frame) + (save_as if save_as.startswith('.') else '.' + save_as)
				tif_stack.save(os.path.join(folder_path, frame_name))
	
	# a small hack to read data_generator internal configuration
	# and know something about '_validation_split' value
	def _is_valid_split_set(self):
		assert self.data_generator is not None
		
		# dataset will be split in two sets whether '_validation_split' > 0.
		return self.data_generator.__dict__['_validation_split'] > 0.

	# it provides an infinite stream of augmented data
	def _generate_data(self, subset, binary_labels=False, **kwargs):
		assert self.data_generator is not None, 'You need to call \'set_data_generator_up\' method'

		kwargs_ = {
			**dict(
				subset=subset,
				shuffle=True if subset == 'training' else False,

				# default
				class_mode=None,
				color_mode='rgb',
				target_size=(256, 256),
				batch_size=1,
				seed=None,
			),
			**kwargs
		}

		# to avoid directory overwriting, just in case it's provided from the user 
		if 'directory' in kwargs_: del kwargs_['directory']

		# setting directories up
		directory_X = os.path.join(self.data_path, 'train')
		directory_y = os.path.join(self.data_path, 'labels')

		# generators
		X = self.data_generator.flow_from_directory(directory_X, **kwargs_)
		y = self.data_generator.flow_from_directory(directory_y, **kwargs_)

		# yeild data from both generators
		for X, y in zip(X, y):

			# transform labels to binary if needed
			if binary_labels:

				# transform output to binary
				y = self._to_binary(y, threshold=0.5)

			yield X, y

	# get traning data through a ImageDataGenerator generators
	# which will handle data augmentation
	def set_generator_up(self, data_augmentation=dict()):
		self.data_generator = ImageDataGenerator(**data_augmentation)

		# inform user about validation split
		if self._is_valid_split_set():
			print('A subset will be used for validation purposes (' + str(data_augmentation['validation_split'] * 100) + '%)')

	# ...
	def generate_train_data(self, binary_labels=False, **kwargs):
		#assert self.data_generator is not None, 'You need to call \'set_data_generator_up\' method'
		#
		#subset = 'training' if self.split_train_for_validation else None
		#
		#train_X = self.data_generator.flow_from_directory(
		#	directory=os.path.join(self.data_path, 'train'),
		#	subset=subset,
		#	class_mode=None,
		#	target_size=image_size,
		#	color_mode=color_mode[0],
		#	batch_size=batch_size,
		#	shuffle=True,
		#	seed=self.seed
		#)
		#train_y = self.data_generator.flow_from_directory(
		#	directory=os.path.join(self.data_path, 'labels'),
		#	class_mode=None,
		#	subset=subset,
		#	target_size=image_size,
		#	color_mode=color_mode[1],
		#	batch_size=batch_size,
		#	shuffle=True,
		#	seed=self.seed
		#)
		#
		## yeild data from both generators
		#for X, y in zip(train_X, train_y):
		#	if classification == 'binary':
		#		y = self._to_binary(y, threshold=0.5)
		#	yield X, y
		assert self.data_generator is not None, 'You need to call \'set_data_generator_up\' method'

		# if validation_split has not been provided,
		# use data just for traning (no subsets)
		subset = 'training' if self._is_valid_split_set() else None

		# generate training data
		return self._generate_data(subset, binary_labels=binary_labels, **kwargs)

	def generate_valid_data(self, binary_labels=False, **kwargs):
		#assert self.data_generator is not None, 'You need to call \'set_data_generator_up\' method'
		#
		#subset = 'validation' if self.split_train_for_validation else None
		#
		#validation_X = self.data_generator.flow_from_directory(
		#	directory=os.path.join(self.data_path, 'train'),
		#	class_mode=None,
		#	subset=subset,
		#	target_size=image_size,
		#	color_mode=color_mode[0],
		#	batch_size=batch_size,
		#	shuffle=False,
		#	seed=self.seed
		#)
		#validation_y = self.data_generator.flow_from_directory(
		#	directory=os.path.join(self.data_path, 'labels'),
		#	class_mode=None,
		#	subset=subset,
		#	target_size=image_size,
		#	color_mode=color_mode[1],
		#	batch_size=batch_size,
		#	shuffle=False,
		#	seed=self.seed
		#)
		#
		## yeild data from both generators
		#for X, y in zip(validation_X, validation_y):
		#	if classification == 'binary':
		#		y = self._to_binary(y, threshold=0.5)
		#	yield X, y

		# mandatory check about validation_split configuration in data_generator
		assert self._is_valid_split_set(), 'You should\'ve set \'validation_split\' in \'set_data_generator_up\' method. All the data were used for training purposes.'  

		# it generates validation data
		return self._generate_data('validation', binary_labels=binary_labels, **kwargs)

	# TODO
	def load_test_data(self):
		pass