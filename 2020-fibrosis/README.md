
### setup environment with docker
```
docker build -t tf2 .

docker run --gpus all -it -v /mnt/hd0/data/kaggle/2020-fibrosis:/kaggle/input/osic-pulmonary-fibrosis-progression -v ${PWD}:/opt -w /opt -p 8888:8888 -u $(id -u):$(id -g)  tf2 /bin/bash
```
#### start notebook
```
jupyter notebook --ip=0.0.0.0 --port=8888
```

