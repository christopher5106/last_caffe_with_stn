# Spatial Transformer Networks and LSTM RNN in Caffe

## Spatial Transformer Networks

Spatial Transformer Networks from https://github.com/daerduoCarey/SpatialTransformerLayer with an updated (last) version of Caffe and some code cleaning and corrections.

Compile Caffe following my tutorial on [iOS](http://christopher5106.github.io/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-mac-osx.html) or [Ubuntu](http://christopher5106.github.io/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html).

My tutorial about [Spatial Transformer Networks](http://christopher5106.github.io/big/data/2016/04/18/spatial-transformer-layers-caffe-tensorflow.html).

Commands to build a Docker :

    docker build -f docker/standalone/cpu/Dockerfile  -t caffe/stn:v1 .

or pull it directly :

    docker pull christopher5106/caffe:stn

## LSTM RNN

LSTM from https://github.com/junhyukoh/caffe-lstm

My tutorial about [LSTM in Caffe](http://christopher5106.github.io/deep/learning/2016/06/07/recurrent-neural-net-with-Caffe.html).
