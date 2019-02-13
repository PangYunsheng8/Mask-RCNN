This is the first version for mask rcnn. This branch is modified from google 
object detection api. The software requriment is same with the original repo, 
    eg: tensorflow >= 1.9, pycocotool etc
There still some bug in generate tf-record when using multiprocessing.

# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim
*   Jupyter notebook
*   Matplotlib
*   Tensorflow (>=1.9.0)
*   Cython
*   contextlib2
*   cocoapi

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
```

Alternatively, users can install dependencies using pip:

``` bash
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

<!-- common_typos_disable -->
**Note**: sometimes "sudo apt-get install protobuf-compiler" will install
Protobuf 3+ versions for you and some users have issues when using 3.5.
If that is your case, try the [manual](#Manual-protobuf-compiler-installation-and-usage) installation.

## COCO API installation

Download the
[cocoapi](https://github.com/cocodataset/cocoapi) and
copy the pycocotools subfolder to the tensorflow/models/research directory if
you are interested in using COCO evaluation metrics. The default metrics are
based on those used in Pascal VOC evaluation. To use the COCO object detection
metrics add `metrics_set: "coco_detection_metrics"` to the `eval_config` message
in the config file. To use the COCO instance segmentation metrics add
`metrics_set: "coco_mask_metrics"` to the `eval_config` message in the config
file.

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_maskrcnn>/
```

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the mask_rcnn directory:


``` bash
# From path_to_maskrcnn/
protoc object_detection/protos/*.proto --python_out=.
```

**Note**: If you're getting errors while compiling, you might be using an incompatible protobuf compiler. If that's the case, use the following manual installation

## Manual protobuf-compiler installation and usage

**If you are on linux:**

Download and install the 3.0 release of protoc, then unzip the file.

```bash
# From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

Run the compilation process again, but use the downloaded version of protoc

```bash
# From path_to_maskrcnn/
./bin/protoc object_detection/protos/*.proto --python_out=.
```

**If you are on MacOS:**

If you have homebrew, download and install the protobuf with
```brew install protobuf```

Alternately, run:
```PROTOC_ZIP=protoc-3.3.0-osx-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
rm -f $PROTOC_ZIP
```

Run the compilation process again:

``` bash
# From mask_rcnn/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the mask_rcnn/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
mask_rcnn/:


``` bash
# From mask_rcnn/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
tensorflow/models/research on your system.

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```

#  Training the Maskrcnn
    Step 1: generate mask from xml annotation
        Follow the repo Anno_parser to generate mask from xml file
    Step 2: convert to tf-record
        Convert the Oxford pet dataset to TFRecord for object_detection.
    
    See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
         Cats and Dogs
         IEEE Conference on Computer Vision and Pattern Recognition, 2012
         http://www.robots.ox.ac.uk/~vgg/data/pets/
    
    Example usage:
        python object_detection/dataset_tools/create_pet_tf_record.py \
            --data_dir=/home/user/pet \
            --output_dir=/home/user/pet/output
    
    flags:
    
    object_detection/dataset_tools/create_mask_tf_record.py:
      --anno_dir: Root directory to xml annotations directory
        (default: '')
      --data_dir: Root directory to raw image dataset.
        (default: '')
      --[no]faces_only: If True, generates bounding boxes for pet faces.  Otherwise
        generates bounding boxes (as well as segmentations for full pet bodies).
        Note that in the latter case, the resulting files are much larger.
        (default: 'true')
      --label_map_path: Path to label map proto
        (default: 'data/pet_label_map.pbtxt')
      --mask_type: How to represent instance segmentation masks. Options are "png"
        or "numerical".
        (default: 'png')
      --[no]multi: If True, use multiprocessing tocreate tf record.
        (default: 'false')
      --num_shards: Number of TFRecord shards
        (default: '10')
        (an integer)
      --output_dir: Path to directory to output TFRecords.
        (default: '')
    
    Try --helpfull to get a list of all flags.
    Step 3: config your pipeline_config file, here are some maskrcnn sample configs 
        in directory object_detection/samples/configs
    step 4: train the model
        bash train_only.sh [gpu-id] [PIPELINE_CONFIG_PATH] [MODEL_DIR] [NUM_TRAIN_STEPS]
    step 5: evaluate the model
        bash val.sh [PIPELINE_CONFIG_PATH] [MODEL_DIR]
