"""
Quantize trained SSD face detector to OpenVINO INT8 (OpenVINO 2024).
"""

import argparse
import importlib
import logging
from pathlib import Path
from typing import List
import traceback
import sys
import types

import numpy as np
import torch

from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.yolo_dataset import YOLODataset
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.data_preprocessing import TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.utils import box_utils


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVINO 2024 INT8 PTQ for SSD face detector")
    parser.add_argument("--model_path", default="models/RFB-Epoch-204-mAP-0.99768739-ValLoss-0.00246579.pth", type=str, help="Path to trained .pth model")
    parser.add_argument("--label_path", default="models/voc-model-labels.txt", type=str, help="Class labels file")
    parser.add_argument("--net_type", default="RFB", choices=["RFB", "slim"], help="Network type")
    parser.add_argument("--input_size", default=720, type=int, help="Model input size")
    parser.add_argument("--output_dir", default="models/openvino_int8", type=str, help="Output directory")
    parser.add_argument("--onnx_path", default="", type=str, help="Optional ONNX output path")
    parser.add_argument("--dataset_type", default="yolo", choices=["yolo", "voc"], help="Validation dataset type")
    parser.add_argument("--yolo_data_yaml", default="/mnt/c/Users/snooker/xanylabeling_data/trainer/ultralytics/datasets/detect/data_20260325_174511_allballs/data.yaml", type=str, help="YOLO data.yaml path")
    parser.add_argument("--voc_val_dir", default="", type=str, help="VOC validation dataset path")
    parser.add_argument("--calib_subset_size", default=300, type=int, help="Calibration sample count")
    parser.add_argument("--val_max_samples", default=500, type=int, help="Max validation samples for mAP compare")
    parser.add_argument("--prob_threshold", default=0.01, type=float, help="Score threshold for eval")
    parser.add_argument("--iou_threshold", default=0.45, type=float, help="NMS IoU threshold")
    parser.add_argument("--candidate_size", default=1000, type=int, help="NMS candidate size")
    parser.add_argument("--device", default="CPU", type=str, help="OpenVINO device for eval")
    return parser.parse_args()


def load_yolo_data_yaml(yaml_path: str):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_class_names(label_path: str) -> List[str]:
    with open(label_path, "r", encoding="utf-8") as f:
        return [name.strip() for name in f.readlines() if name.strip()]


def build_pytorch_model(net_type: str, num_classes: int, model_path: str):
    if net_type == "RFB":
        net = create_Mb_Tiny_RFB_fd(num_classes, is_test=True, device="cpu")
    else:
        net = create_mb_tiny_fd(num_classes, is_test=True, device="cpu")
    net.load(model_path)
    net.eval()
    return net


def export_to_onnx(net, input_size: int, onnx_path: Path):
    img_size_dict = {
        128: [128, 96], 160: [160, 120], 320: [320, 240], 480: [480, 360],
        640: [640, 480], 720: [720, 540], 960: [960, 720], 1280: [1280, 960],
    }
    h, w = img_size_dict[input_size][1], img_size_dict[input_size][0]
    dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net, dummy_input, str(onnx_path), verbose=False, opset_version=13,
        input_names=["input"], output_names=["scores", "boxes"],
    )
    logging.info("ONNX exported: %s", onnx_path)


def build_val_dataset(args, test_transform):
    if args.dataset_type == "yolo":
        if not args.yolo_data_yaml:
            raise ValueError("--yolo_data_yaml is required for yolo dataset_type")
        cfg = load_yolo_data_yaml(args.yolo_data_yaml)
        root = cfg.get("path")
        return YOLODataset(root, transform=test_transform, target_transform=None, is_test=True, split="val", data_config=cfg)
    if not args.voc_val_dir:
        raise ValueError("--voc_val_dir is required for voc dataset_type")
    return VOCDataset(args.voc_val_dir, transform=test_transform, target_transform=None, is_test=True)


def get_image_and_gt(dataset, idx: int):
    image_id = dataset.ids[idx]
    image = dataset._read_image(image_id)
    if isinstance(dataset, VOCDataset):
        gt_boxes, gt_labels, is_difficult = dataset._get_annotation(image_id)
        if not dataset.keep_difficult:
            gt_boxes = gt_boxes[is_difficult == 0]
            gt_labels = gt_labels[is_difficult == 0]
    else:
        gt_boxes, gt_labels = dataset._get_annotation(image_id, image)
    if len(gt_boxes) == 0:
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.zeros((0,), dtype=np.int64)
    return image, gt_boxes.astype(np.float32), gt_labels.astype(np.int64)


def nms_decode(scores, boxes, width, height, prob_threshold, iou_threshold, candidate_size):
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, scores.shape[1]):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.from_numpy(np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1))
        box_probs = box_utils.nms(
            box_probs, nms_method=None, score_threshold=prob_threshold, iou_threshold=iou_threshold,
            sigma=0.5, top_k=-1, candidate_size=candidate_size,
        ).numpy()
        if box_probs.size == 0:
            continue
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    picked_box_probs = np.concatenate(picked_box_probs, axis=0)
    picked_box_probs[:, [0, 2]] *= width
    picked_box_probs[:, [1, 3]] *= height
    return picked_box_probs[:, :4].astype(np.float32), np.array(picked_labels, dtype=np.int64), picked_box_probs[:, 4].astype(np.float32)


def bbox_iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)))
    x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    y2 = np.minimum(box1[:, None, 3], box2[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / np.clip(union, 1e-6, None)


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def calculate_map(predictions, gts, num_classes, iou_thresh=0.5):
    aps = []
    for cls_id in range(1, num_classes):
        tp, fp, scores = [], [], []
        total_gt = 0
        for i in range(len(predictions)):
            pred_boxes, pred_labels, pred_scores = predictions[i]
            gt_boxes, gt_labels = gts[i]
            pred_mask = pred_labels == cls_id
            gt_mask = gt_labels == cls_id
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]
            gt_boxes = gt_boxes[gt_mask]
            total_gt += len(gt_boxes)
            if len(pred_boxes) == 0:
                continue
            scores.extend(pred_scores.tolist())
            if len(gt_boxes) == 0:
                fp.extend([1] * len(pred_boxes))
                tp.extend([0] * len(pred_boxes))
                continue
            ious = bbox_iou(pred_boxes, gt_boxes)
            matched = set()
            for j in range(len(pred_boxes)):
                iou = np.max(ious[j])
                idx_gt = int(np.argmax(ious[j]))
                if iou >= iou_thresh and idx_gt not in matched:
                    tp.append(1)
                    fp.append(0)
                    matched.add(idx_gt)
                else:
                    tp.append(0)
                    fp.append(1)
        if total_gt == 0:
            continue
        if len(tp) == 0:
            aps.append(0.0)
            continue
        scores = np.array(scores)
        order = np.argsort(-scores)
        tp = np.cumsum(np.array(tp)[order])
        fp = np.cumsum(np.array(fp)[order])
        recall = tp / (total_gt + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        aps.append(float(compute_ap(recall, precision)))
    return float(np.mean(aps)) if aps else 0.0


def _extract_scores_boxes(result):
    tensors = [np.array(v) for v in result.values()]
    scores = None
    boxes = None
    for arr in tensors:
        if arr.ndim == 3 and arr.shape[-1] == 4 and boxes is None:
            boxes = arr
        elif arr.ndim == 3 and arr.shape[-1] != 4 and scores is None:
            scores = arr
    if scores is None or boxes is None:
        raise RuntimeError(
            f"Cannot identify scores/boxes from model outputs. "
            f"Output shapes: {[tuple(t.shape) for t in tensors]}"
        )
    return scores[0], boxes[0]


def evaluate_map(compiled_model, dataset, test_transform, max_samples, prob_threshold, iou_threshold, candidate_size):
    predictions = []
    gts = []
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    for idx in range(total):
        image, gt_boxes, gt_labels = get_image_and_gt(dataset, idx)
        h, w = image.shape[:2]
        img_tensor, _, _ = test_transform(image, gt_boxes.copy(), gt_labels.copy())
        inp = np.expand_dims(img_tensor.numpy(), axis=0).astype(np.float32)
        result = compiled_model([inp])
        scores, boxes = _extract_scores_boxes(result)
        pred_boxes, pred_labels, pred_scores = nms_decode(scores, boxes, w, h, prob_threshold, iou_threshold, candidate_size)
        predictions.append((pred_boxes, pred_labels, pred_scores))
        gts.append((gt_boxes, gt_labels))
    return calculate_map(predictions, gts, len(dataset.class_names), iou_thresh=0.5)


class CalibrationDataset:
    def __init__(self, dataset, test_transform, subset_size):
        self.dataset = dataset
        self.test_transform = test_transform
        self.total = min(len(dataset), subset_size) if subset_size > 0 else len(dataset)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        image, gt_boxes, gt_labels = get_image_and_gt(self.dataset, idx)
        img_tensor, _, _ = self.test_transform(image, gt_boxes, gt_labels)
        return {"input": np.expand_dims(img_tensor.numpy(), axis=0).astype(np.float32)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    define_img_size(args.input_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx_path) if args.onnx_path else output_dir / "model.onnx"
    fp32_xml = output_dir / "model_fp32.xml"
    int8_xml = output_dir / "model_int8.xml"

    class_names = read_class_names(args.label_path)
    net = build_pytorch_model(args.net_type, len(class_names), args.model_path)

    logging.info("Step1: export ONNX")
    export_to_onnx(net, args.input_size, onnx_path)

    import openvino as ov
    try:
        from openvino import convert_model
    except ImportError:
        from openvino.tools.ovc import convert_model

    # NNCF/OpenVINO compatibility shim:
    # some NNCF versions expect symbols on top-level openvino module.
    try:
        import openvino.runtime as ov_runtime
        if not hasattr(ov, "Node") and hasattr(ov_runtime, "Node"):
            ov.Node = ov_runtime.Node  # type: ignore[attr-defined]
        if not hasattr(ov, "Output") and hasattr(ov_runtime, "Output"):
            ov.Output = ov_runtime.Output  # type: ignore[attr-defined]
        if not hasattr(ov, "Input") and hasattr(ov_runtime, "Input"):
            ov.Input = ov_runtime.Input  # type: ignore[attr-defined]
        if not hasattr(ov, "Type") and hasattr(ov_runtime, "Type"):
            ov.Type = ov_runtime.Type  # type: ignore[attr-defined]
        if not hasattr(ov, "PartialShape") and hasattr(ov_runtime, "PartialShape"):
            ov.PartialShape = ov_runtime.PartialShape  # type: ignore[attr-defined]
        if not hasattr(ov, "Shape") and hasattr(ov_runtime, "Shape"):
            ov.Shape = ov_runtime.Shape  # type: ignore[attr-defined]
    except Exception:
        # Keep going; if symbols still mismatch, downstream error message is clearer.
        pass

    # Some NNCF builds import `openvino.op`, which does not exist in newer OpenVINO wheels.
    # Register a compatible alias from available opset module.
    try:
        opset_module = None
        for opset_name in ["opset13", "opset12", "opset11", "opset10", "opset9", "opset8"]:
            if hasattr(ov_runtime, opset_name):
                opset_module = getattr(ov_runtime, opset_name)
                break
        if opset_module is not None:
            # Some NNCF versions expect CamelCase op constructors (e.g. op.Constant),
            # while newer OpenVINO opset modules expose lowercase helpers (e.g. op.constant).
            camel_aliases = {
                "Constant": "constant",
                "Convert": "convert",
                "Concat": "concat",
                "Reshape": "reshape",
                "Transpose": "transpose",
                "MatMul": "matmul",
                "Multiply": "multiply",
                "Subtract": "subtract",
                "Add": "add",
            }
            for camel_name, lower_name in camel_aliases.items():
                if not hasattr(opset_module, camel_name) and hasattr(opset_module, lower_name):
                    setattr(opset_module, camel_name, getattr(opset_module, lower_name))

            if "openvino.op" not in sys.modules:
                sys.modules["openvino.op"] = opset_module
            # Some NNCF versions import a fixed `openvino.opset13`.
            if "openvino.opset13" not in sys.modules:
                sys.modules["openvino.opset13"] = opset_module
    except Exception:
        pass

    # Some NNCF builds import `openvino.utils.node_factory`, while newer layouts keep it
    # under `openvino.runtime.utils.node_factory`.
    try:
        if "openvino.utils.node_factory" not in sys.modules:
            runtime_nf = importlib.import_module("openvino.runtime.utils.node_factory")
            utils_mod = sys.modules.get("openvino.utils")
            if utils_mod is None:
                utils_mod = types.ModuleType("openvino.utils")
                sys.modules["openvino.utils"] = utils_mod
            sys.modules["openvino.utils.node_factory"] = runtime_nf
            setattr(utils_mod, "node_factory", runtime_nf)
            if hasattr(runtime_nf, "NodeFactory"):
                setattr(utils_mod, "NodeFactory", runtime_nf.NodeFactory)
    except Exception:
        pass

    # Some NNCF builds reference low-bit/float8 OpenVINO types that may be absent
    # in certain OpenVINO 2024 Python packages. Patch them to nearest available types.
    try:
        missing_type_fallbacks = {
            "nf4": "f16",
            "f4e2m1": "f16",
            "f8e8m0": "f16",
            "f8e4m3": "f16",
            "f8e5m2": "f16",
            "u4": "u8",
            "i4": "i8",
        }
        type_obj = ov.Type
        setattr_failed = False
        for missing_name, fallback_name in missing_type_fallbacks.items():
            if not hasattr(type_obj, missing_name) and hasattr(type_obj, fallback_name):
                try:
                    setattr(type_obj, missing_name, getattr(type_obj, fallback_name))
                except Exception:
                    setattr_failed = True

        if setattr_failed:
            class _OVTypeProxy:
                pass

            proxy = _OVTypeProxy()
            base_names = [
                "f16", "bf16", "f32", "f64",
                "i8", "i32", "i64", "u8", "u16", "u32",
                "u4", "i4", "nf4", "f4e2m1", "f8e8m0", "f8e4m3", "f8e5m2",
            ]
            for name in base_names:
                if hasattr(type_obj, name):
                    setattr(proxy, name, getattr(type_obj, name))
            for missing_name, fallback_name in missing_type_fallbacks.items():
                if not hasattr(proxy, missing_name) and hasattr(proxy, fallback_name):
                    setattr(proxy, missing_name, getattr(proxy, fallback_name))
            ov.Type = proxy  # type: ignore[attr-defined]
    except Exception:
        pass

    logging.info("Step2: ONNX -> OpenVINO FP32")
    ov_model_fp32 = convert_model(str(onnx_path))
    ov.save_model(ov_model_fp32, str(fp32_xml), compress_to_fp16=False)

    test_transform = TestTransform([args.input_size, int(args.input_size * 3 / 4)], np.array([127, 127, 127]), 128.0)
    val_dataset = build_val_dataset(args, test_transform)
    calib_dataset = CalibrationDataset(val_dataset, test_transform, args.calib_subset_size)

    logging.info("Step3: PTQ INT8 (preset=MIXED, accuracy-first)")
    # Compatibility workaround for old PyTorch (e.g. torch==1.12) with newer NNCF.
    # Some NNCF builds reference newer dtypes during import.
    dtype_fallbacks = {
        "uint16": torch.int32,
        "uint32": torch.int64,
        "uint64": torch.int64,
        "float8_e4m3fn": torch.float16,
        "float8_e5m2": torch.float16,
    }
    for name, fallback in dtype_fallbacks.items():
        if not hasattr(torch, name):
            setattr(torch, name, fallback)
    try:
        import nncf
    except Exception as exc:
        raise RuntimeError(
            "Failed to import nncf. This is usually a torch/nncf version mismatch. "
            "For torch1.12, try reinstalling a compatible nncf version in current env, "
            "or run quantization in a separate newer PyTorch env.\n"
            f"torch version: {torch.__version__}\n"
            f"original import error: {exc}\n"
            f"traceback:\n{traceback.format_exc()}"
        ) from exc
    # NNCF API differs across versions:
    # - newer: transform_fn
    # - older: transform_func
    # - some versions: no transform arg needed
    try:
        nncf_calib = nncf.Dataset(calib_dataset, transform_fn=lambda item: item)
    except TypeError:
        try:
            nncf_calib = nncf.Dataset(calib_dataset, transform_func=lambda item: item)
        except TypeError:
            nncf_calib = nncf.Dataset(calib_dataset)
    ov_model_int8 = nncf.quantize(
        ov_model_fp32,
        calibration_dataset=nncf_calib,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=min(len(calib_dataset), args.calib_subset_size),
        fast_bias_correction=True,
    )
    ov.save_model(ov_model_int8, str(int8_xml), compress_to_fp16=False)

    logging.info("Step4: evaluate FP32 vs INT8 mAP")
    core = ov.Core()
    compiled_fp32 = core.compile_model(ov_model_fp32, args.device)
    compiled_int8 = core.compile_model(ov_model_int8, args.device)
    fp32_map = evaluate_map(
        compiled_fp32, val_dataset, test_transform, args.val_max_samples,
        args.prob_threshold, args.iou_threshold, args.candidate_size,
    )
    int8_map = evaluate_map(
        compiled_int8, val_dataset, test_transform, args.val_max_samples,
        args.prob_threshold, args.iou_threshold, args.candidate_size,
    )
    logging.info("FP32 mAP: %.6f", fp32_map)
    logging.info("INT8 mAP: %.6f", int8_map)
    logging.info("mAP drop: %.6f", fp32_map - int8_map)
    logging.info("FP32 IR: %s", fp32_xml)
    logging.info("INT8 IR: %s", int8_xml)


if __name__ == "__main__":
    main()
