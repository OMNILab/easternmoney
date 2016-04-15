#!/bin/bash

export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.75
#export KERAS_BACKEND=tensorflow
python easternmoney_review_lstm.py &> output.log &

