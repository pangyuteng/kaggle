import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp_name', type=str,default='0')
args = parser.parse_args()
exp_name = args.exp_name
exp_folder = f"/kaggle/temp/exp/{exp_name}"
model_path = os.path.join(exp_folder,"model.h5")
history_path = os.path.join(exp_folder,'history.yaml')

import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from prepare import (
    raw_list_path,
    MyDataGenerator,
)
from model import get_my_model
from train import my_loss

with open(raw_list_path,'r') as f:
    raw_list = json.loads(f.read())

X_tt, X_test, y_tt, y_test = train_test_split(raw_list,raw_list,test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tt,y_tt,test_size=0.25, random_state=42)

print('6:2:2')
print(len(X_train))
print(len(X_val))
print(len(X_test))

# Build model
model = get_my_model()
opt = keras.optimizers.Adam()
model.compile(optimizer=opt, loss=my_loss())
model.load_weights(model_path)

batch_size = 32
train_gen = MyDataGenerator(X_train,batch_size=batch_size)
val_gen = MyDataGenerator(X_val,batch_size=batch_size)
test_gen = MyDataGenerator(X_test,batch_size=batch_size)

mydict = {}
for kind,gen in [
    ('train',train_gen),
    ('val',val_gen),
    ('test',test_gen),
    ]:
    out = model.evaluate(gen)
    mydict[kind]=out

with open('test.json','w') as f:
    f.write(json.dumps(mydict))

print(mydict)