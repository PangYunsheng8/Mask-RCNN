#!/bin/bash
# Exporting a trained model for inference

if [ $# -ne 3 ]
then
    echo "Usage $0 [PIPELINE_CONFIG_PATH] [TRAINED_CKPT_PREFIX] [EXPORT_DIR]"
    exit 1
fi

INPUT_TYPE=image_tensor
# path to pipeline config file
PIPELINE_CONFIG_PATH=$1
# path to model.ckpt
TRAINED_CKPT_PREFIX=$2
# path to folder that will be used for export 
EXPORT_DIR=$3

python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
