from types import GeneratorType
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras import layers

# model structure taken from u-net-release/phseg_v5-train.prototxt
# The authors eschew padding and as result propose cropping in the upsampling
# layers. Here instead we use padding to avoid the need for cropping.

# TODO: remove classification bool since everything depends on the output_shape:
# output_shape == 1 then sigmoid/binary_crossentropy; otherwise, softmax/categorical_crossentropy
class UNet(object):
	def __init__(self, input_shape, output_shape, classification=None):
		self.model = None
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.classification = classification
		self.classification_opts = {
			'binary': {
				'output_activation': 'sigmoid',
				'optimizer': 'adam',
				'loss': 'binary_crossentropy'
			},
			'multi': {
				'output_activation': 'softmax',
				'optimizer': 'adam', # TEST SGD
				'loss': 'categorical_crossentropy' # TEST dice_coeff
			}
		}

		# this would propably be useful in case of any misspelling...
		assert classification in self.classification_opts.keys()

		# initialize model
		self._build_model()

	## Layers ##
	## The names of the functions below refer to the original paper dictionary ##
	
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

		# apply Conv2D N times if needed
		for i in range(repeat):
			x = layers.Conv2D(n_filters, **kwargs_)(x)
			
			# https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
			if batch_norm: x = layers.BatchNormalization(momentum=0.99)(x)
			
			# activate
			x = layers.Activation('relu')(x)
		return x

	# max-pool
	def _max_pool(self, x, pool_size=(2, 2)):
		return layers.MaxPooling2D(pool_size=pool_size)(x)

	# concatenation of one or more layers
	def _copy_and_crop(self, x=[], axis=3):
		return layers.concatenate(x, axis=axis)

	# https://stats.stackexchange.com/questions/252810/in-cnn-are-upsampling-and-transpose-convolution-the-same
	def _up_conv(self, x, n_filters, **kwargs):
		kwargs_ = {
			**dict(
				kernel_size=(3, 3),
				strides=(2, 2),
				padding='same'
			),
			**kwargs
		}
		return layers.Conv2DTranspose(n_filters, **kwargs_)(x)

	# input
	# e.g. (n_samples, width, height, channels)
	# (3, 128, 128, 3) --> a batch of three 128x128 images in RGB
	def _inputs(self):
		return layers.Input(self.input_shape)

	# output
	# https://stats.stackexchange.com/questions/246287/regarding-the-output-format-for-semantic-segmentation
	# TODO: refactor this method after testing the pixel-wise softmax
	def _outputs(self, x):

		# we use sigmoid since we're working with B/W masks
		if self.classification == 'binary':
			x = self._conv(x, self.output_shape, kernel_size=(1, 1))

		# TEST: pixelwise probability vector -> (batch_size, rows*cols, n_classes)
		else:
			_, n_rows, n_cols, _ = self.input_shape
			x = self._conv(x, self.output_shape, kernel_size=1, strides=1)
			x = layers.Reshape((self.output_shape, n_rows * n_cols))(x)
			x = layers.Permute((2,1))(x)
		
		# use the most appropriate activation function based on the classification task
		activation_kind = self.classification_opts[self.classification]['output_activation']
		x = layers.Activation(activation_kind)(x)
		return x

	# assemble the U-shaped architecture
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

		# output
		outputs = self._outputs(x)
					
		self.model = Model(inputs=[inputs], outputs=[outputs])

	# compile model
	# TODO: add more verbosity
	def _compile_model(self):
		assert self.model is not None

		optimizer = self.classification_opts[self.classification]['optimizer']
		loss = self.classification_opts[self.classification]['loss']
		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

	# set useful callbacks
	def _set_callbacks(self, model_checkpoint=False, early_stopping=False, reduce_lr=False):
		callbacks_ = []

		if model_checkpoint:
			checkpoint = callbacks.ModelCheckpoint(
				filepath='./checkpoints/model-{epoch:02d}.ckpt', 
				save_weights_only=True,
				save_freq='epoch',
				verbose=1
			)
			callbacks_.append(checkpoint)

		# TODO: early_stopping
		if early_stopping:
			stopping = callbacks.EarlyStopping(
				monitor='val_loss',
				patience=2,
				verbose=1
			)
			callbacks_.append(stopping)

		# TODO: gradually decrease learning rate
		if reduce_lr:
			reduc_lr = callbacks.ReduceLROnPlateau(
				monitor='val_acc',
				factor=0.1,
				patience=1,
				min_lr=.001,
				verbose=1
			)
			callbacks_.append(reduc_lr)

		return callbacks_

	# Fit data to the model. Note that data could be a generator too.
	def train(self, data, val_data=None, epochs=1, steps_per_epoch=None, model_checkpoint=False, early_stopping=False, reduce_lr=False):

		# set optimizers, loss function and so forth...
		self._compile_model()

		callbacks_ = self._set_callbacks(model_checkpoint, early_stopping)

		# fit data 
		history = self.model.fit(
			data,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=val_data,

			# we assume that the 20% of steps_per_epoch is good enough for each validation round
			validation_steps=int(steps_per_epoch * 0.20) if val_data is not None else None,
			callbacks=callbacks_,
			verbose=1
		)
		return history

	def predict(self, data, threshold=None, **kwargs):
		predictions = self.model.predict(data, **kwargs)

		# round predictions 
		if threshold is not None:
			return np.where(predictions > threshold, 1, 0)
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

	def get_model(self):
		return self.model

	def get_weights(self):
		return self.model.get_weights()

	def summary(self):
		self.model.summary()
