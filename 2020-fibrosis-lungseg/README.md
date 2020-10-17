### setup environment with docker
```
docker build -t tf2 .

docker run --gpus all -it -v /mnt/hd0/data/kaggle/2020-fibrosis:/kaggle/input/osic-pulmonary-fibrosis-progression -v /mnt/hd0/data/kaggle/temp:/kaggle/temp -v ${PWD}:/opt -w /opt -p 8889:8888 -p 6006:6006 -u $(id -u):$(id -g)  tf2 /bin/bash
```

#### prepare data,train model,evaluate model,visualize

```
python prepare.py
CUDA_VISIBLE_DEVICES=0 python train.py 0
python eval.py
python viz.py
```

#### start tensorboard
docker exec -it ${running-container}
tensorboard --logdir=logs --bind_all

#### start notebook
docker exec -it ${running-container}
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
