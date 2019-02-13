#!/bin/bash
# From the tensorflow/models/research/ directory

if [ $# -ne 4 ]
then
	echo "usage $0 [gpu-id] [PIPELINE_CONFIG_PATH] [MODEL_DIR] [NUM_TRAIN_STEPS]"
	exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
# PIPELINE_CONFIG_PATH=`pwd`/object_detection/samples/configs/mask_rcnn_resnet50_mengniu.config
# MODEL_DIR=/home/yangqihong/mask_mengniu_resnet50/
# NUM_TRAIN_STEPS=150000
PIPELINE_CONFIG_PATH=$2
MODEL_DIR=$3
NUM_TRAIN_STEPS=$4
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_train.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

