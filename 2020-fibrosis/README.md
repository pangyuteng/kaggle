
### setup environment with docker
```
docker build -t tf2 .

docker run --gpus all -it -v /mnt/hd0/data/kaggle/2020-fibrosis:/kaggle/input/osic-pulmonary-fibrosis-progression -v ${PWD}:/opt -w /opt -p 8888:8888 -u $(id -u):$(id -g)  tf2 /bin/bash
```
#### start notebook
```
jupyter notebook --ip=0.0.0.0 --port=8888
```


#### scripts used here is to facilitate competition submission.

```

link https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/discussion/164883
author Julia Elliott

Per the Competition Rules, publicly and freely available external data is permitted, if it is available for use that includes research or academic purposes. If you are a winner, your solution must also be released under the stated open source license, per the rules' winner license requirements.

All external data must be posted to this forum thread no later than the Entry Deadline (one week before competition close).

Once someone posts an external dataset to this thread, you do not need to re-post it if you are using the same one.

You only need to declare the original dataset used; you do not need to declare re-labeled or augmented or otherwise processed versions of datasets. Pre-trained models can be declared, as well; however, models resulting from your own original work, that you have trained yourself offline do not need to be shared/declared.

```

Per the above, no delaring this repo, furhter its 1 week past entry deadline...too late for delaring anyways.

