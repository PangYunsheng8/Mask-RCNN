# encoding=utf-8

import os
import re
import json
import argparse
from utils import crop_resize_addAlpha
import tensorflow as tf

build_tensor_info = tf.saved_model.utils.build_tensor_info

parser = argparse.ArgumentParser()
parser.add_argument("--detection_model_path",
                    required=True,
                    help="detection")
parser.add_argument("--classify_model_path",
                    default=[], nargs="+",
                    help="classification model paths.")
parser.add_argument("--export_dir",
                    required=True,
                    help="Exported directory")
parser.add_argument("--version",
                    default=0,
                    help="Set exported model version. If don't set, version will increment.")
parser.add_argument("--clc_2_dtc_map",
                    default="",
                    help="a json format file which contains classification class to detection class map")
args = parser.parse_args()


# os.environ['TF_ENABLE_WHILE_V2'] = '1'
# os.environ['TF_ENABLE_COND_V2'] = '1'


def load_graph(graph_def_path, input_map=None):
    graph = tf.get_default_graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_def_path, "rb") as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='', input_map=input_map)
    return graph


def load_frozen_detection_graph(graph_def_path, input_map,
                                input_output_tensor_prefix=""):
    # add prefix to input tensors
    if isinstance(input_map, dict) and input_output_tensor_prefix:
        input_map_ = {input_output_tensor_prefix + k: v for k, v in input_map.items()}
        input_map = input_map_

    detection_graph = load_graph(graph_def_path, input_map=input_map)
    # get_detection_graph_inputs_outputs
    input_images = detection_graph.get_tensor_by_name(input_output_tensor_prefix + "image_tensor:0")
    detection_boxes = detection_graph.get_tensor_by_name(input_output_tensor_prefix + 'detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name(input_output_tensor_prefix + 'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(input_output_tensor_prefix + 'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(input_output_tensor_prefix + 'num_detections:0')
    masks = detection_graph.get_tensor_by_name(input_output_tensor_prefix + "detection_masks:0")
    outputs = {"boxes": detection_boxes, "scores": detection_scores,
               "classes": detection_classes, "num_detections": num_detections,
               "masks": masks}

    return detection_graph, input_images, outputs


def load_frozen_classify_graph(graph_def_path, input_map,
                               input_output_tensor_prefix=""):
    # add prefix to input tensors
    if isinstance(input_map, dict) and input_output_tensor_prefix:
        input_map_ = {input_output_tensor_prefix + k: v for k, v in input_map.items()}
        input_map = input_map_

    classify_graph = load_graph(graph_def_path, input_map=input_map)
    # get_classify_graph_inputs_outputs
    input_images = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "input_images:0")
    predict = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "predict:0")
    logits = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "logits:0")
    features = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "features:0")
    outputs = {"logits": logits, "predict": predict, "features": features}

    return classify_graph, input_images, outputs


def get_input_output_tensor_prefix(graph_def_path):
    prefix = ""
    info_path = os.path.join(os.path.dirname(graph_def_path), "model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)
        prefix = model_info.get("input_output_tensor_prefix", "")
    return prefix


def build_merged_detection_classification(detection_graph_def_path, classify_graph_def_path,
                                          detection_class_index_map,
                                          classification_class_index_map,
                                          detection_score_threshold=0.5,
                                          classification_class_2_detection_class_map=None):
    graph = tf.Graph()
    with graph.as_default():
        encoded_image = tf.placeholder(tf.string, name="detection_encoded_image")
        image_tensor = tf.image.decode_image(encoded_image, channels=3)
        shp = tf.shape(image_tensor)
        input_image_tensor = tf.reshape(image_tensor, tf.stack([-1, shp[-3], shp[-2], 3]), "detection_image_tensor")
        # detection phase
        prefix = get_input_output_tensor_prefix(detection_graph_def_path)
        detection_graph, d_input_images, d_outputs = \
            load_frozen_detection_graph(detection_graph_def_path,
                                        input_map={"image_tensor:0": input_image_tensor},
                                        input_output_tensor_prefix=prefix)
        d_outputs["classes"] = tf.cast(d_outputs["classes"], tf.int64)

        # drop boxes which's score is under a given threshold
        valid = tf.greater_equal(tf.reshape(d_outputs["scores"], [-1]),
                                 detection_score_threshold)
        # num = tf.argmin(tf.cast(valid, tf.float32), output_type=tf.int32)
        num = tf.reduce_sum(tf.cast(valid, tf.int32))
        d_boxes = tf.slice(d_outputs["boxes"], tf.stack([0, 0, 0]), tf.stack([-1, num, -1]),
                           name="detection_only_boxes")
        d_scores = tf.slice(d_outputs["scores"], tf.stack([0, 0]), tf.stack([-1, num]))
        d_scores = tf.squeeze(d_scores, axis=0, name="detection_only_scores")
        d_classes = tf.slice(d_outputs["classes"], tf.stack([0, 0]), tf.stack([-1, num]))
        d_classes = tf.squeeze(d_classes, axis=0, name="detection_only_classes")
        d_masks = tf.slice(d_outputs['masks'], [0, 0, 0, 0], [-1, num, -1, -1])
        d_masks = tf.squeeze(d_masks, 0, name='detection_only_masks')

        # make sure classify model has at least one patch to classify
        num_for_classify = tf.maximum(1, num)
        d_boxes_for_clf = tf.slice(d_outputs["boxes"],
                                   [0, 0, 0],
                                   [-1, num_for_classify, -1])
        print('d_boxes_for_clf:', d_boxes_for_clf.shape)
        d_masks_for_clf = tf.slice(d_outputs['masks'], [0, 0, 0, 0],
                                   [-1, num_for_classify, -1, -1])
        detect_images = crop_resize_addAlpha(input_image_tensor,
                                             d_boxes_for_clf,
                                             d_masks_for_clf,
                                             crop_size=[224, 224])
        detect_images = tf.cast(detect_images, tf.uint8)
        shp = tf.shape(detect_images)
        patch_images = tf.reshape(detect_images,
                                  [shp[0], shp[1], shp[2], 4],
                                  name='classify_image_tensor')

        # classification phase
        if not isinstance(classify_graph_def_path, list):
            classify_graph_def_path = [classify_graph_def_path]
        # embedding
        all_probability = []
        all_features = []
        for cls_graph_def_path in classify_graph_def_path:
            prefix = get_input_output_tensor_prefix(cls_graph_def_path)
            classify_graph_, c_input_images_, c_outputs_ = \
                load_frozen_classify_graph(cls_graph_def_path,
                                           input_map={"input_images": patch_images},
                                           input_output_tensor_prefix=prefix)
            probability = tf.nn.softmax(c_outputs_["logits"])
            all_probability.append(probability)
            all_features.append(c_outputs_["features"])
        probability = tf.reduce_mean(tf.stack(all_probability, axis=0), axis=0)
        features = tf.reduce_mean(tf.stack(all_features, axis=0), axis=0)
        c_outputs = {"scores": tf.reduce_max(probability, axis=-1),
                     "predict": tf.argmax(probability, axis=-1),
                     "features": features}

        c_scores = tf.identity(c_outputs["scores"], name="classify_scores")
        c_classes = tf.identity(c_outputs["predict"], name="classify_classes")
        c_features = tf.identity(c_outputs["features"], name="features")
        c_outputs["scores"] = tf.slice(c_outputs["scores"], [0], [num])
        c_outputs["predict"] = tf.slice(c_outputs["predict"], [0], [num])
        c_outputs["features"] = tf.slice(c_outputs["features"], [0, 0], [num, -1])
        scores = tf.identity(c_outputs["scores"], name="detection_classify_scores")
        classes = tf.identity(c_outputs["predict"], name="detection_classify_classes")
        features = tf.identity(c_outputs["features"], name="detection_classify_features")

        # translate class index to name
        detection_index_class_map = {v: k for k, v in detection_class_index_map.items()}
        detection_class_names = [detection_index_class_map.get(i, "Unkown")
                                 for i in range(1 + max(detection_index_class_map.keys()))]
        detection_class_names = tf.constant(detection_class_names)

        classification_index_class_map = {v: k for k, v in classification_class_index_map.items()}
        classification_class_names = [classification_index_class_map.get(i, "Unkown")
                                      for i in range(1 + max(classification_index_class_map.keys()))]
        classification_class_names = tf.constant(classification_class_names)

        c_classes_names = tf.gather(classification_class_names, c_classes)
        d_classes_names = tf.gather(detection_class_names, d_classes)
        classes_names = tf.gather(classification_class_names, classes)

        inputs = {
            "detection_encoded_image": encoded_image,
            "detection_image_tensor": input_image_tensor,
            "classify_image_tensor": patch_images
        }
        outputs = {
            "classify_scores": c_scores,
            "classify_classes": c_classes,
            "classify_classes_names": c_classes_names,
            "classify_features": c_features,
            "detection_only_boxes": d_boxes,
            "detection_only_scores": d_scores,
            "detection_only_classes": d_classes,
            "detection_only_classes_names": d_classes_names,
            "detection_only_masks": d_masks,
            "detection_classify_scores": scores,
            "detection_classify_classes": classes,
            "detection_classify_classes_names": classes_names,
            "detection_classify_features": features
        }

    return graph, inputs, outputs


def save_detection_classify_model(graph, inputs, outputs, saved_model_path):
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
    with tf.Session(graph=graph) as sess:
        # build tensor info of inputs outputs
        for key in inputs:
            inputs[key] = build_tensor_info(inputs[key])
        for key in outputs:
            if isinstance(outputs[key], tf.Tensor):
                outputs[key] = build_tensor_info(outputs[key])
            elif isinstance(outputs[key], dict):
                outputs[key] = {k: build_tensor_info(v) for k, v in outputs[key].items()}
            else:
                raise ValueError("Unexpected outputs type, only accept tf.Tensor or a dict as string to tf.Tensor map.")

        # detection only
        detection_only_inputs_tensor_info = {
            'inputs': inputs["detection_image_tensor"]
        }
        detection_only_outputs_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_only_scores"],
            "classes": outputs["detection_only_classes"]
        }
        detection_only_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_only_inputs_tensor_info,
                outputs=detection_only_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # classify only
        classify_only_inputs_tensor_info = {
            'inputs': inputs["classify_image_tensor"]
        }
        classify_only_outputs_tensor_info = {
            "scores": outputs["classify_scores"],
            "classes": outputs["classify_classes"],
            "features": outputs["classify_features"]
        }
        classify_only_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=classify_only_inputs_tensor_info,
                outputs=classify_only_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # merged detection classify
        detection_classify_inputs_tensor_info = {
            'inputs': inputs["detection_image_tensor"]
        }
        detection_classify_outputs_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_classify_scores"],
            "classes": outputs["detection_classify_classes"],
            "features": outputs["detection_classify_features"]
        }
        detection_classify_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_classify_inputs_tensor_info,
                outputs=detection_classify_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        # =============== return class names ============== #
        # detection only
        detection_only_outputs_with_names_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_only_scores"],
            "classes": outputs["detection_only_classes_names"]
        }
        detection_only_with_names_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_only_inputs_tensor_info,
                outputs=detection_only_outputs_with_names_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # classify only
        classify_only_outputs_with_names_tensor_info = {
            "scores": outputs["classify_scores"],
            "classes": outputs["classify_classes_names"]
        }
        classify_only_with_names_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=classify_only_inputs_tensor_info,
                outputs=classify_only_outputs_with_names_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # merged detection classify
        detection_classify_outputs_with_names_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_classify_scores"],
            "classes": outputs["detection_classify_classes_names"]
        }
        detection_classify_with_names_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_classify_inputs_tensor_info,
                outputs=detection_classify_outputs_with_names_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        # ==================== with features ================ #
        # classify only
        classify_only_outputs_with_features_tensor_info = {
            "scores": outputs["classify_scores"],
            "classes": outputs["classify_classes"],
            "features": outputs["classify_features"]
        }
        classify_only_with_features_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=classify_only_inputs_tensor_info,
                outputs=classify_only_outputs_with_features_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # merged detection classify
        detection_classify_inputs_with_features_tensor_info = {
            'inputs': inputs["detection_image_tensor"]
        }
        detection_classify_outputs_with_features_tensor_info = {
            "boxes": outputs["detection_only_boxes"],
            "scores": outputs["detection_classify_scores"],
            "classes": outputs["detection_classify_classes"],
            "features": outputs["detection_classify_features"]
        }
        detection_classify_with_features_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=detection_classify_inputs_tensor_info,
                outputs=detection_classify_outputs_with_features_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "detection_only": detection_only_signature,
                "classify_only": classify_only_signature,
                "detection_classify": detection_classify_signature,
                "detection_only_with_names": detection_only_with_names_signature,
                "classify_only_with_names": classify_only_with_names_signature,
                "detection_classify_with_names": detection_classify_with_names_signature,
                "classify_only_with_features": classify_only_with_features_signature,
                "detection_classify_with_features": detection_classify_with_features_signature
            }
        )
        builder.save()


def _completing_model_path(model_path_or_dir):
    return os.path.join(model_path_or_dir, "frozen_inference_graph.pb") if \
        os.path.isdir(model_path_or_dir) else model_path_or_dir


def main(_):
    assert args.detection_model_path != ""
    assert args.classify_model_path != []
    assert args.export_dir != ""

    saved_model_path = os.path.join(args.export_dir, str(args.version))
    if args.version == 0 and os.path.exists(saved_model_path):
        versions = [int(x) for x in os.listdir(args.export_dir) if os.path.isdir(os.path.join(args.export_dir, x))
                    and re.match("^\d+$", x)]
        args.version = 1 + max(versions)
        saved_model_path = os.path.join(args.export_dir, str(args.version))
        print("Auto choose saved model version: %s" % args.version)

    if os.path.exists(saved_model_path):
        raise Exception("The saved model version(%s) is already exists, please choose a bigger one." % args.version)

    args.detection_model_path = _completing_model_path(args.detection_model_path)
    args.classify_model_path = [_completing_model_path(x) for x in args.classify_model_path]

    # check model compatible
    prefixes = [get_input_output_tensor_prefix(x) for x in args.classify_model_path]
    t = {}
    for pfx, pth in zip(prefixes, args.classify_model_path):
        if pfx not in t:
            t[pfx] = pth
        else:
            raise Exception("The model input output tensor prefixes are same (both is '%s') "
                            "of model '%s' and '%s'. Please make them be "
                            "different." % (pfx, t[pfx], pth))

    detection_label_index_map_path = \
        os.path.join(os.path.dirname(args.detection_model_path), "label_index.map")
    classify_label_index_map_path = \
        os.path.join(os.path.dirname(args.classify_model_path[0]), "label_index.map")
    with open(detection_label_index_map_path) as f:
        detection_label_index_map = json.load(f)
        if isinstance(detection_label_index_map, list):
            detection_label_index_map = {x['class']: x['id'] for x in detection_label_index_map}
    with open(classify_label_index_map_path) as f:
        classify_label_index_map = json.load(f)
        classify_label_index_map_src = classify_label_index_map
        if isinstance(classify_label_index_map, list):
            classify_label_index_map = {x['class']: x['id'] for x in classify_label_index_map}

    clc_2_dtc_map = None
    if args.clc_2_dtc_map != "":
        with open(args.clc_2_dtc_map) as f:
            clc_2_dtc_map = json.load(f)
        t0 = set(clc_2_dtc_map.keys())
        t1 = set(classify_label_index_map.keys())
        assert len(t0.intersection(t1)) == len(t0) <= len(t1)
        t0 = set(clc_2_dtc_map.values())
        t1 = set(detection_label_index_map.keys())
        assert len(t0.intersection(t1)) == len(t0) <= len(t1)

    graph, inputs, outputs = build_merged_detection_classification(
        args.detection_model_path,
        args.classify_model_path,
        detection_label_index_map,
        classify_label_index_map,
        classification_class_2_detection_class_map=clc_2_dtc_map)

    serialized_graph = graph.as_graph_def().SerializeToString()
    save_detection_classify_model(graph, inputs, outputs, saved_model_path)
    with tf.gfile.GFile(os.path.join(saved_model_path, "frozen_inference_graph.pb"), "wb") as f:
        f.write(serialized_graph)
    with open(os.path.join(saved_model_path, "label_index.map"), "w") as f:
        json.dump(classify_label_index_map_src, f, indent=4)
    with open(os.path.join(saved_model_path, "model_info.json"), "w") as f:
        model_info = {
            "detection_model_path": os.path.abspath(args.detection_model_path),
            "classify_model_path": [os.path.abspath(x) for x in args.classify_model_path],
            "clc_2_dtc_map": os.path.join(args.clc_2_dtc_map)
        }
        json.dump(model_info, f, indent=4)
    print("Exported model to: %s" % saved_model_path)


if __name__ == '__main__':
    tf.app.run()
