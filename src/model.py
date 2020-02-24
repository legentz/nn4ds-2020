import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras import layers

## Layers ##
## The names of the functions below refer to the original paper dictionary ##
class UNet(object):
	def __init__(self, input_shape, output_shape):
		self.model = None
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.compile_opt = {
			'optimizer': 'adam',
			'loss': 'binary_crossentropy',
			'metrics': ['accuracy']
		}
		self.training_ops = {
			#'callbacks': ['model_checkpoint', 'early_stopping', 'reduce_lr_on_plateau'],
			'valid_steps_perc': 0.2,
			'is_valid_data_available': False
		}  

		# initialize model
		self._build_model()
	
	# dropout
	def _dropout(self, x, value):
		return layers.Dropout(value)(x)

	# down-convolution (ReLU activated) with optional batch norm
	def _conv(self, x, n_filters, repeat=1, batch_norm=False, **kwargs):
		kwargs_ = {
			# https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference 
			**dict(
				kernel_size=3,
				kernel_initializer='glorot_normal',
				padding='same'
			),
			**kwargs
		}

		# apply Conv2D n times if needed
		for i in range(repeat):
			x = layers.Conv2D(n_filters, **kwargs_)(x)
			
			# https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
			if batch_norm: x = layers.BatchNormalization(momentum=0.99)(x)
			
			# activate
			x = layers.Activation('relu')(x)
		return x

	# max-pool layer
	def _max_pool(self, x, pool_size=(2, 2)):
		return layers.MaxPooling2D(pool_size=pool_size)(x)

	# concatenation layer
	def _copy_and_crop(self, x=[], axis=3):
		return layers.concatenate(x, axis=axis)

	# upsampling layer
	# https://stats.stackexchange.com/questions/252810/in-cnn-are-upsampling-and-transpose-convolution-the-same
	def _up_conv(self, x, n_filters, **kwargs):
		kwargs_ = {
			**dict(
				kernel_size=(3, 3),
				strides=(2, 2),

				# Here instead we use padding to avoid the need for cropping.
				padding='same'
			),
			**kwargs
		}
		return layers.Conv2DTranspose(n_filters, **kwargs_)(x)

	# input layer
	# ex. (n_samples, width, height, channels) -> channel_last format
	def _inputs(self):
		return layers.Input(self.input_shape)

	# output layer
	# output for binary classification only (no softmax)
	def _outputs(self, x):
		x = self._conv(x, self.output_shape, kernel_size=(1, 1))

		# we use sigmoid since we're working with B/W masks
		x = layers.Activation('sigmoid')(x)
		return x

	# assemble the U-shaped architecture
	# overall structure taken from the official u-net-release/phseg_v5-train.prototxt
	# ref.: https://arxiv.org/pdf/1505.04597.pdf
	def _build_model(self):

		# input
		inputs = self._inputs()
		
		# left
		x_64 = self._conv(inputs, 64, repeat=2, batch_norm=True)
		x = self._max_pool(x_64)
		x_128 = self._conv(x, 128, repeat=2, batch_norm=True)
		x = self._max_pool(x_128)
		x_256 = self._conv(x, 256, repeat=2, batch_norm=True)
		x = self._max_pool(x_256)
		x_512 = self._conv(x, 512, repeat=2, batch_norm=True)
		x = self._max_pool(x_512)

		# dropout will be applied during training only
		# Keras takes the merit
		self._dropout(x, 0.5)

		# bottleneck
		x = self._conv(x, 1024, repeat=2, batch_norm=True)

		# dropout will be applied during training only
		# Keras takes the merit
		self._dropout(x, 0.5)

		# right
		x = self._up_conv(x, 512)
		x = self._copy_and_crop([x, x_512])
		x = self._conv(x, 512, repeat=2, batch_norm=True)
		x = self._up_conv(x, 256)
		x = self._copy_and_crop([x, x_256])
		x = self._conv(x, 256, repeat=2, batch_norm=True)
		x = self._up_conv(x, 128)
		x = self._copy_and_crop([x, x_128])
		x = self._conv(x, 128, repeat=2, batch_norm=True)
		x = self._up_conv(x, 64)
		x = self._copy_and_crop([x, x_64])
		x = self._conv(x, 64, repeat=2, batch_norm=True)

		# sigmoidal output
		outputs = self._outputs(x)
		
		# set model
		self.model = Model(inputs=[inputs], outputs=[outputs])

	# compile
	def _compile_model(self):
		assert self.model is not None

		# compile model for binary classification based on sigmoidal output
		self.model.compile(
			optimizer=self.compile_opt['optimizer'],
			loss=self.compile_opt['loss'],
			metrics=self.compile_opt['metrics']
		)

	# set useful callbacks for traning
	def _set_callbacks(self, cbks):
		callbacks_ = []

		# save weights of each checkpoint to easily restore later
		if 'model_checkpoint' in cbks:
			m_heckpoint = callbacks.ModelCheckpoint(
				filepath='./checkpoints/model-{epoch:02d}.ckpt', 
				save_weights_only=True,
				save_freq='epoch',
				verbose=1
			)
			callbacks_.append(m_heckpoint)

		# early stopping the training phase when loss value stops deacreasing
		if 'early_stopping' in cbks:

			# monitor different quantities whethere we have any validation round or not
			if self.training_ops['is_valid_data_available']:
				monitor = 'val_loss'
			else:
				monitor = 'loss'

			e_stopping = callbacks.EarlyStopping(
				monitor=monitor,
				patience=2,
				verbose=1
			)
			callbacks_.append(e_stopping)
			#else: print('WARNING: Cannot use early_stopping callbacks since no validation data has been provided')

		# gradually decrease learning rate when the model stops learning
		if 'reduce_lr_on_plateau' in cbks:

			# monitor different quantities whethere we have any validation round or not
			if self.training_ops['is_valid_data_available']:
				monitor = 'val_loss'
			else:
				monitor = 'loss'

			reduce_lr = callbacks.ReduceLROnPlateau(
				monitor=monitor,
				factor=0.1,
				patience=1,
				min_lr=.000001,
				verbose=1
			)
			callbacks_.append(reduce_lr)

		return callbacks_

	# Fit data to the model
	# NOTE: data could be a generator too.
	def train(self, data, val_data=None, epochs=1, steps_per_epoch=None, callbacks=[]): #model_checkpoint=False, early_stopping=False, reduce_lr=False

		# in case we have any validation data 
		if val_data is not None:

			# this flag helps the model remember whether we need to perfom any validation rounds
			self.training_ops['is_valid_data_available'] = True 

			# compute how many steps should we perform for each validation round
			# we assume that the 20% of steps_per_epoch is good enough for each validation round
			validation_steps = int(steps_per_epoch * self.training_ops['valid_steps_perc'])
		else:
			self.training_ops['is_valid_data_available'] = False 

		# set optimizers, loss function and so forth...
		self._compile_model()

		# set useful callbacks (ex. EarlyStopping)
		callbacks_ = self._set_callbacks(callbacks)

		# fit data 
		history = self.model.fit(
			data,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=val_data,
			validation_steps=validation_steps,
			callbacks=callbacks_,
			verbose=1
		)
		return history

	# predict (default: from a generator) and round the resulting preds (if requested)
	# use the kwargs to provide batch_size whether not using a generator
	# TODO: predict on batch to keep the labels
	def predict(self, data, threshold=None, **kwargs):
		predictions = self.model.predict(data, **kwargs)

		# round sigmoidal predictions 
		if threshold is not None:
			return predictions, np.where(predictions < threshold, 0, 1)
		return predictions

	# resume weights from a specific checkpoint file (or .h5)
	# or the latest one from the provided directory
	def load_weights(self, to_restore, checkpoint=False):
		assert os.path.exists(to_restore)

		# if a directory is provided, load the last checkpoint
		if os.path.isdir(to_restore) and checkpoint:
			to_restore = tf.train.latest_checkpoint(to_restore)

		# restore weights
		self.model.load_weights(to_restore)

	# Apart from the checkpoints, this method allows to save the entire mode
	# which could be restored the same way as checkpoints
	def save_weights(self, h5_path, overwrite=False):
		assert self.model is not None

		if os.path.exists(h5_path):
			if not overwrite:
				raise('Cannot save model: ' + h5_path + ' already exists')
		self.model.save_weights(h5_path)


	# get the Keras model
	def get_model(self):
		return self.model

	# show the overall picture of UNet architecture
	def summary(self):
		self.model.summary()
