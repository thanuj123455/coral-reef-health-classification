from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam


def build_vgg19(input_shape=(128, 128, 3), learning_rate=0.001):
      """Build VGG19 transfer learning model.

          Uses ImageNet pre-trained weights with frozen base layers
              and a custom classification head for binary output.

                  Args:
                          input_shape: Tuple (H, W, C) for input images.
                                  learning_rate: Adam optimizer learning rate.

                                      Returns:
                                              Compiled Keras Sequential model.
                                                  """
      base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
      for layer in base_model.layers:
                layer.trainable = False

      model = Sequential([
          base_model,
          Flatten(),
          Dense(256, activation='relu'),
          Dense(1, activation='sigmoid')
      ])
      model.compile(
          optimizer=Adam(learning_rate=learning_rate),
          loss='binary_crossentropy',
          metrics=['accuracy']
      )
      return model
  
