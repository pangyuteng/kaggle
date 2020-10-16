import os
import sys
import yaml
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from prepare import (
    raw_list_path,
    MyDataGenerator,
)

with open(raw_list_path,'r') as f:
    raw_list = json.loads(f.read())

X_tt, X_test, y_tt, y_test = train_test_split(raw_list,raw_list,test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tt,y_tt,test_size=0.25, random_state=42)

print('6:2:2')
print(len(X_train))
print(len(X_val))
print(len(X_test))


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--exp_name', type=str,default='0')
    args = parser.parse_args()
    exp_name = args.exp_name

    exp_folder = f"/kaggle/temp/exp/{exp_name}"
    os.makedirs(exp_folder,exist_ok=True)
    model_path = os.path.join(exp_folder,"model.h5")
    history_path = os.path.join(exp_folder,'history.yaml')

    keras.backend.clear_session()

    if exp_name == '0':
        from model import MyModelBuilder,batch_size,epochs
    else:
        raise NotImplementedError()
    
    builder = MyModelBuilder()
    model = builder.build()

    # Train the model, doing validation at the end of each epoch.
    train_gen = MyDataGenerator(X_train,batch_size=batch_size)
    val_gen = MyDataGenerator(X_val)
    log_dir = f'logs/{exp_name}'

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        builder.get_image_summary_callback(val_gen,log_dir),
        keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
        )
    ]

    history = model.fit(train_gen,epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    with open(history_path,'w') as f:
        f.write(yaml.dump(history.history))