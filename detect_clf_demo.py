import numpy as np
import tensorflow as tf
import cv2
import argparse
import utils
import pdb

parser = argparse.ArgumentParser(description="Demo for testing model that merged by "
                                             "detection model and classify model")
parser.add_argument("--image", metavar="path", type=str,
                    help="the path to the image directory")
parser.add_argument("--model_path", metavar='Path',
                    help="path to the frozen pb model")


def load_graph(graph_def_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_def_path, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph


args = parser.parse_args()
graph = load_graph(args.model_path)


def main():
    with graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        keys = ["classify_image_tensor", "classify_classes", "classify_scores"]
        output_dict = {}
        for key in keys:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                output_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        input_tensor = tf.get_default_graph().get_tensor_by_name('detection_image_tensor:0')
        for img_path in utils.read_file_from_directory(args.image, postfix=".jpg"):
            im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            with tf.Session() as sess:
                tensor_dict = sess.run(output_dict, feed_dict={input_tensor: np.expand_dims(im, axis=0)})
                print("classify input:", tensor_dict['classify_image_tensor'].shape)
                print("classes:", tensor_dict['classify_classes'])
                print("scores:", tensor_dict['classify_scores'])


if __name__ == '__main__':
    main()
