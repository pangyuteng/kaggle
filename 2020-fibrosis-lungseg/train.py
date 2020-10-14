import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from prepare import (
    raw_list_path,
    MyDataGenerator,
)
from model import get_my_model

with open(raw_list_path,'r') as f:
    raw_list = json.loads(f.read())

X_tt, X_test, y_tt, y_test = train_test_split(raw_list,raw_list,test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tt,y_tt,test_size=0.25, random_state=42)

print('6:2:2')
print(len(X_train))
print(len(X_val))
print(len(X_test))

# training hyper parameters
lr = 0.01
batch_size = 8
epochs = 5

def bc_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def my_loss():
    def loss(y_true, y_pred):
        return bc_loss(y_true, y_pred)
    return loss


if __name__ == '__main__':
    keras.backend.clear_session()

    model = get_my_model()

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=my_loss())

    callbacks = [
        keras.callbacks.ModelCheckpoint("/kaggle/temp/mymodel.h5", save_best_only=True),
        keras.callbacks.TensorBoard(
            log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
        )
    ]

    # Train the model, doing validation at the end of each epoch.
    train_gen = MyDataGenerator(X_train,batch_size=batch_size)
    val_gen = MyDataGenerator(X_val)
    history = model.fit(train_gen,epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    with open('history.yaml','w') as f:
        f.write(yaml.dump(history.history))