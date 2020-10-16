import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e','--exp_name', type=str,default='0')
args = parser.parse_args()
exp_name = args.exp_name
exp_folder = f"/kaggle/temp/exp/{exp_name}"
model_path = os.path.join(exp_folder,"model.h5")

if exp_name == '0':
    from model import MyModelBuilder,batch_size,epochs
else:
    raise NotImplementedError()

builder = MyModelBuilder()
model = builder.build()
model.load_weights(model_path)


import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from jinja2 import Environment

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


batch_size = 32
test_gen = MyDataGenerator(X_test,batch_size=batch_size)
myx,myy = test_gen[0]
myy_hat = model.predict(myx)
print(myy_hat.shape)
os.makedirs('html',exist_ok=True)
mylist = []
for x in range(batch_size):
    fname = f'html/x_viz_test{x}.png'
    plt.figure(figsize=(20,10))
    plt.subplot(131)
    plt.imshow(myx[x,:].squeeze(),cmap='gray')
    plt.subplot(132)
    plt.imshow(myy[x,:].squeeze(),cmap='gray')
    plt.subplot(133)
    plt.imshow(myy_hat[x,:].squeeze(),cmap='gray')
    plt.savefig(fname)
    plt.close()
    mylist.append(fname)

# https://gist.github.com/wrunk/1317933
HTML = '''
<html>
<head>
</head>
<body>
{% for item in mylist %}
<img src="{{item}}" height=400px>
<hr>
{% endfor %}
</body>
</html>
'''
with open('html/x_viz_test.html','w') as f:
    html = Environment().from_string(HTML).render(mylist=mylist)
    f.write(html)
