import tensorflow as tf
from model import UNet
from data import EMData

EPOCHS = 5

def data():
	emdata = EMData(data_path='./../_data/em_segmentation')
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
		batch_size=2,
		seed=666
	)
	return train_data

def run_tf(train_data):

	# this is required in order to make @tf.function work properly
	tf.config.experimental_run_functions_eagerly(True)

	unet = UNet()
	loss_object = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam()

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	@tf.function
	def train_step(X, y):
		with tf.GradientTape() as tape:
			predictions = unet(X, training=True)
			loss = loss_object(y, predictions)
			print('loss', loss)
		gradients = tape.gradient(loss, unet.trainable_variables)
		print('gradients', gradients)
		print('trainable_variables', unet.trainable_variables)
		optimizer.apply_gradients(zip(gradients, unet.trainable_variables))

		train_loss(loss)
		train_accuracy(y, predictions)

	for epoch in range(1, EPOCHS + 1):

		# reset the metrics at the start of the next epoch
		train_loss.reset_states()
		train_accuracy.reset_states()

		# iterate all over the dataset 
		for X, y in train_data:

			# traning function
			train_step(X, y)

			# feedback to user
			print('Epoch {}, Loss: {}, Accuracy: {}'.format(epoch, train_loss.result(), train_accuracy.result()*100))

def run_k(train_data):
	unet = UNet()
	unet.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
	unet.fit_generator(train_data, steps_per_epoch=300)

# main
if __name__ == '__main__':
	train_data = data()
	run_tf(train_data)
	#run_k(train_data)