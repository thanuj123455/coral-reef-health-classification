import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Input,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply
)


def cbam_block(input_tensor, reduction_ratio=8):
      """CBAM: Convolutional Block Attention Module."""
      channel = input_tensor.shape[-1]
      avg_pool = GlobalAveragePooling2D()(input_tensor)
      max_pool = GlobalMaxPooling2D()(input_tensor)
      shared_dense = Dense(channel // reduction_ratio, activation='relu', use_bias=False)
      avg_out = shared_dense(avg_pool)
      max_out = shared_dense(max_pool)
      channel_attention = Add()([avg_out, max_out])
      channel_attention = Dense(channel, activation='sigmoid')(channel_attention)
      channel_attention = Multiply()([input_tensor, channel_attention])
      avg_spatial = tf.keras.layers.Lambda(
          lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_attention)
      max_spatial = tf.keras.layers.Lambda(
          lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_attention)
      spatial_attention = Conv2D(
          1, kernel_size=7, activation='sigmoid', padding='same'
      )(Add()([avg_spatial, max_spatial]))
      return Multiply()([channel_attention, spatial_attention])


def build_cnn_with_cbam(input_shape=(128, 128, 3)):
      """Build Custom CNN with CBAM attention blocks."""
      inputs = Input(shape=input_shape)
      x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
      x = tf.keras.layers.MaxPooling2D((2, 2))(x)
      x = cbam_block(x)
      x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
      x = tf.keras.layers.MaxPooling2D((2, 2))(x)
      x = cbam_block(x)
      x = Flatten()(x)
      x = Dense(128, activation='relu')(x)
      outputs = Dense(1, activation='sigmoid')(x)
      model = Model(inputs, outputs)
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      return model
  
