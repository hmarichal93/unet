import tensorflow as tf

from keras.UNET import UNET

def build_dataset():
    pass

def main():
    # Example usage of the UNET model
    model = UNET(padding='same', batchnorm=True, classes=1)
    model.build(input_shape=(None, 512, 512, 3))
    model.summary()
    trainds, evalds = build_dataset()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(trainds,
              validation_data=evalds,
              epochs=NUM_EVALS,
              steps_per_epoch=steps_per_epoch)