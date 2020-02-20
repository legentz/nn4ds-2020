import os, sys, re
import numpy as np 
from PIL import Image

class Iterator(object):

	@staticmethod
	def imgs_from_folder(directory=None, normalize=False, output_shape=None):
		assert os.path.exists(directory)

		list_dir = os.listdir(directory)
		list_dir.sort(key=lambda f: int(re.sub('\D', '', f)))

		for img in list_dir:

			try:
				img = Image.open(os.path.join(directory, img))
			except:
				print('WARNING:', sys.exc_info()[0])
				continue

			img = np.asarray(img)

			if normalize:
				img = img / img.max()

			if output_shape is not None:
				img = img.reshape(output_shape)

			yield(img)