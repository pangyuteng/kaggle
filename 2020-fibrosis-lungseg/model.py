import os
import random
import numpy as np
import tensorflow as tf
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(42)

from tensorflow import keras
from tensorflow.keras import layers

# training hyper parameters
def bc_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def my_loss():
    def loss(y_true, y_pred):
        return bc_loss(y_true, y_pred)
    return loss

lr = 0.01
batch_size = 8
epochs = 100

class ImageSummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, logsdir, builder):
        super(ImageSummaryCallback, self).__init__()
        self.logsdir = logsdir
        self.val_data = val_data
        self.file_writer = tf.summary.create_file_writer(self.logsdir)
        self.builder = builder

    def on_batch_end(self, batch, logs):
        
        batch = 0
        imgs = self.val_data[batch][0]
        masks = self.val_data[batch][1]
        masks_hat = self.builder.model(imgs, training=False)

        #convert float to uint8, shift range to 0-255
        imgs -= tf.reduce_min(imgs)
        imgs *= 255 / tf.reduce_max(imgs)
        imgs = tf.cast(imgs, tf.uint8)

        masks *= 255
        masks = tf.cast(masks, tf.uint8)

        masks_hat *= 255
        masks_hat = tf.cast(masks_hat, tf.uint8)

        with self.file_writer.as_default():

            #for ix in range(masks_hat.shape[0]):    
            ix = 0
            img = imgs[ix,:]
            mask = masks[ix,:]
            masks_hat = masks_hat[ix,:]

            img_tensor = tf.expand_dims(img, 0)
            mask_tensor = tf.expand_dims(mask, 0)
            masks_hat_tensor = tf.expand_dims(masks_hat, 0)

            #only post 1 out of every x images to tensorboard
            if (ix % 4) == 0:
                tf.summary.image(f'x/image{ix}', img_tensor, step=batch)
                tf.summary.image(f'y/mask{ix}', mask_tensor, step=batch)
                tf.summary.image(f'y_hat/mask{ix}', masks_hat_tensor, step=batch)

            self.file_writer.flush()
# https://keras.io/examples/vision/oxford_pets_image_segmentation/


class MyModelBuilder():
    def __init__(self):
        pass
    def build(self,
            compile=True,
            img_size=(512,512),
            num_classes=1,
            ):
        
        inputs = keras.Input(shape=img_size + (1,))

        ### [First half of the network: downsampling inputs] ###
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside residual
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x) # <--- ... 
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
                
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        for filters in [256, 128, 64, 32]:
            
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual
        
        # Add a per-pixel classification layer
        outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)
        
        # Define the model
        model = keras.Model(inputs, outputs)
        if compile:
            opt = keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=opt, loss=my_loss())
        
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        return model

    def get_image_summary_callback(self,data_generator,log_dir):
        return ImageSummaryCallback(data_generator,log_dir,self)


if __name__ == '__main__':
    #keras.backend.clear_session()
    # Build model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--debug',action='store_true')
    args = parser.parse_args()

    if args.debug:
        model = MyModelBuilder().build(img_size=(28,28))
        model.summary()

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train, X_test = X_train/255, X_test/255
        history = model.fit(x=X_train, y=X_train, batch_size=10,epochs=10, validation_data=(X_test, X_test))

    else:

        model = MyModelBuilder().build()
        model.summary()
