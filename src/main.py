import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from model import UNet
from data import EMData
import matplotlib.pyplot as plt

EPOCHS = 5

def data():
	emdata = EMData(data_path='./../data/em_segmentation')
	#emdata.unpack(overwrite=True, save_as='.png')

	train_data = emdata.load_training_data(
		data_augmentation={
			'rotation_range': 2,
			'width_shift_range': 0.05,
			'height_shift_range': 0.05,
			'brightness_range': [0.8, 1.2],
			'rescale': 1./255,
			'shear_range': 0.05,
			'zoom_range': 0.05,
			'horizontal_flip': True,
			'vertical_flip': True
		},
		batch_size=1,
		seed=666
	)
	return train_data

def run_k(train_data):
	# Loss: https://datascience.stackexchange.com/questions/42599/what-is-the-relationship-between-the-accuracy-and-the-loss-in-deep-learning
	unet = UNet((512, 512, 1), 1).get_model()
	unet.summary()
	#unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	unet.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
	history = unet.fit(train_data, epochs=EPOCHS, steps_per_epoch=100, verbose=1)

	# show some hisory...
	acc = history.history['accuracy']
	loss = history.history['loss']

	plt.figure(figsize=(8, 8))
	plt.plot(range(EPOCHS), acc, label='Training Accuracy')
	plt.plot(range(EPOCHS), loss, label='Training Loss')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')
	plt.savefig('plot.png')

'''
Main
'''
if __name__ == '__main__':
	my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
	tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

	train_data = data()
	run_k(train_data)