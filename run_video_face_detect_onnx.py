"""
This code uses the onnx model to detect faces from live video or cameras.
Timing style aligned with typical C++ pipeline:
preprocess + infer + postprocess
"""

import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
import onnxruntime as ort


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]

    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue

        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)

        box_probs = box_utils.hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )

        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)

    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    return (
        picked_box_probs[:, :4].astype(np.int32),
        np.array(picked_labels),
        picked_box_probs[:, 4],
    )


label_path = "models/voc-model-labels.txt"

onnx_path = "/mnt/e/projs/py_projs/Ultra-Light-Fast-Generic-Face-Detector-1MB_SOD/models/onnx/env_rfb_1280.onnx"

class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

cap = cv2.VideoCapture("/mnt/e/test_video/6/_5.0.7.0_251224_231655-S76球停袋口误判进袋，所有异常/output_stand.mp4")

threshold = 0.7

sum_faces = 0

while True:

    ret, orig_image = cap.read()

    if orig_image is None:
        print("no img")
        break

    t0 = time.time()

    # ----------------------
    # preprocess
    # ----------------------

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1280, 960))

    image_mean = np.array([127, 127, 127])

    image = (image - image_mean) / 128

    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    t1 = time.time()

    # ----------------------
    # inference
    # ----------------------

    confidences, boxes = ort_session.run(None, {input_name: image})

    t2 = time.time()

    # ----------------------
    # postprocess
    # ----------------------

    boxes, labels, probs = predict(
        orig_image.shape[1],
        orig_image.shape[0],
        confidences,
        boxes,
        threshold
    )

    t3 = time.time()

    # ----------------------
    # draw result
    # ----------------------

    for i in range(boxes.shape[0]):

        box = boxes[i, :]

        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        cv2.rectangle(
            orig_image,
            (box[0], box[1]),
            (box[2], box[3]),
            (255, 255, 0),
            4
        )

    sum_faces += boxes.shape[0]

    orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)

    # ----------------------
    # timing print
    # ----------------------

    preprocess_time = (t1 - t0) * 1000
    infer_time = (t2 - t1) * 1000
    postprocess_time = (t3 - t2) * 1000
    total_time = (t3 - t0) * 1000

    print(
        f"preprocess: {preprocess_time:.2f} ms | "
        f"infer: {infer_time:.2f} ms | "
        f"postprocess: {postprocess_time:.2f} ms | "
        f"total: {total_time:.2f} ms"
    )

    cv2.imshow("annotated", orig_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("sum:", sum_faces)