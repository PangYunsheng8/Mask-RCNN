#!/bin/bash
# From the tensorflow/models/research/ directory

if [ $# -ne 2 ]
then
    echo "usage $0 [PIPELINE_CONFIG_PATH] [MODEL_DIR]"
    exit 1
fi

PIPELINE_CONFIG_PATH=$1
MODEL_DIR=$2
NUM_TRAIN_STEPS=150000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --checkpoint_dir=${MODEL_DIR} \
    --alsologtostderr

