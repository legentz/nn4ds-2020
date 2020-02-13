import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Input, concatenate

class UNet(Model):
	def __init__(self):
		super(UNet, self).__init__()

		# dropout
		self.dropout = Dropout(0.5)

		# conv
		conv = lambda n_filters: Conv2D(n_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
		self.filters_2_conv = {k: conv(k) for k in [64, 128, 256, 512, 1024]}
		self.conv = self.__conv_block
		
		# concat
		self.copy_and_crop = lambda x: concatenate(x, axis=3)
		
		# downsample
		self.max_pool = MaxPooling2D(pool_size=(2, 2))
		
		# upsample
		up_conv = lambda n_filters: Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')
		self.filters_2_up_conv = {k: up_conv(k) for k in [64, 128, 256, 512]}
		self.up_conv = self.__up_conv_block

		self.outputs = Conv2D(1, (3, 3), activation='softmax', padding='same', kernel_initializer='he_normal')
	
	def __up_conv_block(self, x, n_filters):
		conv2d_t = self.filters_2_up_conv[n_filters]
		return conv2d_t(x)
	
	def __conv_block(self, x, n_filters, repeat=1):
		for i in range(repeat):
			conv2d = self.filters_2_conv[n_filters]
			x = conv2d(x)
		return x

	def __call__(self, x, training=False):

		# left
		x_64 = self.conv(x, 64, repeat=2)
		x = self.max_pool(x_64)
		x_128 = self.conv(x, 128, repeat=2)
		x = self.max_pool(x_128)
		x_256 = self.conv(x, 256, repeat=2)
		x = self.max_pool(x_256)
		x_512 = self.conv(x, 512, repeat=2)
		x = self.max_pool(x_512)

		# center
		x_1024 = self.conv(x, 1024, repeat=2)
		x = self.up_conv(x_1024, 512)

		# right
		x = self.copy_and_crop([x, x_512])
		x = self.conv(x, 512, repeat=2)
		x = self.up_conv(x, 256)
		x = self.copy_and_crop([x, x_256])
		x = self.conv(x, 256, repeat=2)
		x = self.up_conv(x, 128)
		x = self.copy_and_crop([x, x_128])
		x = self.conv(x, 128, repeat=2)
		x = self.up_conv(x, 64)
		x = self.copy_and_crop([x, x_64])
		x = self.conv(x, 64, repeat=2)

		#if training: x = self.dropout(x) <<< TODO

		# output
		return self.outputs(x)



