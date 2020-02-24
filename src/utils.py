# TODO: better imports... more specific
import os, sys, re, math
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

class Iterator(object):

	@staticmethod
	# TODO: batched data
	def imgs_from_folder(directory=None, normalize=False, output_shape=None):
		assert os.path.exists(directory)

		list_dir = os.listdir(directory)
		list_dir.sort(key=lambda f: int(re.sub('\D', '', f)))

		for img in list_dir:

			# TODO: make this a standalon function under Images utils
			try:
				img = Image.open(os.path.join(directory, img))
			except:
				print('WARNING:', sys.exc_info()[0])
				continue

			img = np.asarray(img)

			if normalize:
				img = img / img.max() # TODO: 255 

			if output_shape is not None:
				img = img.reshape(output_shape)

			yield(img)

class Images(object):

	@staticmethod
	# TODO: automatic rows/cols based on len(arr)
	def plot_arr_to_imgs(arr, n_max=-1, cols=1, cmap=None, figsize=(15, 15)):
		fig = plt.figure(figsize=figsize)

		n_imgs = len(arr)
		rows = math.ceil(n_imgs / cols)

		for i in range(0, cols * rows):

			# load n_max images
			if i >= n_max:
				break

			fig.add_subplot(rows, cols, i + 1)

			if len(arr.shape) == 4:
				img = arr[i, :, :, 0]

			elif len(arr.shape) == 3:
				img = arr[:, :, i]

			elif len(arr.shape) == 2:
				img = arr

			else:
				raise('Bad shaped array provided!')

			# add image to grid
			plt.imshow(img, cmap=cmap)

		# show
		plt.show()

	@staticmethod
	def plot_imgs_from_folder(path, n_max=-1, cols=1, cmap=None, figsize=(15, 15)):

		# path should exist
		assert os.path.exists(path)

		imgs = os.listdir(path)
		imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
		n_imgs = len(imgs) 
		rows = math.ceil(n_imgs / cols) 

		fig = plt.figure(figsize=figsize)

		for i in range(0, n_imgs):

			# load n_max images
			if i >= n_max:
				break

			fig.add_subplot(rows, cols, i + 1)

			# TODO: make this a standalon function under Images utils
			try:
				img = Image.open(os.path.join(path, imgs[i]))
			except:
				print('WARNING:', sys.exc_info()[0])
				continue

			img = np.asarray(img)

			# add image to grid
			plt.imshow(img, cmap=cmap)

		# show
		plt.show()

	@staticmethod
	def save_as_imgs(arr, save_as='.png'):
		pass



