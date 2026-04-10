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
import time

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
    parser.add_argument("--max_map_drop", default=0.005, type=float, help="Max allowed mAP drop (FP32-INT8)")
    parser.add_argument("--strict_accuracy", action="store_true", help="Fail and skip saving INT8 when mAP drop exceeds threshold")
    parser.add_argument("--benchmark_samples", default=200, type=int, help="Frames/samples for latency benchmark")
    parser.add_argument("--warmup_samples", default=20, type=int, help="Warmup samples before benchmark")
    parser.add_argument("--quant_mode", default="auto", choices=["auto", "performance", "mixed"], help="INT8 quantization mode")
    parser.add_argument("--benchmark_infer_only", action="store_true", help="Benchmark only inference time (exclude preprocess)")
    parser.add_argument("--fp32_perf_hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT"], help="OpenVINO PERFORMANCE_HINT for FP32 model")
    parser.add_argument("--int8_perf_hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT"], help="OpenVINO PERFORMANCE_HINT for INT8 model")
    parser.add_argument("--fp32_num_streams", default="1", type=str, help="FP32 NUM_STREAMS (e.g. 1, 2, AUTO)")
    parser.add_argument("--int8_num_streams", default="1", type=str, help="INT8 NUM_STREAMS (e.g. 1, 2, AUTO)")
    parser.add_argument("--fp32_threads", default=0, type=int, help="FP32 INFERENCE_NUM_THREADS (0 means default)")
    parser.add_argument("--int8_threads", default=0, type=int, help="INT8 INFERENCE_NUM_THREADS (0 means default)")
    parser.add_argument(
        "--speed_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prioritize speed for INT8 selection. If true, default to performance mode and skip accuracy gating unless --strict_accuracy is set.",
    )
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


def benchmark_latency(compiled_model, dataset, test_transform, max_samples, warmup_samples):
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    warmup = min(total, max(0, warmup_samples))
    infer_times = []

    for idx in range(total):
        image, gt_boxes, gt_labels = get_image_and_gt(dataset, idx)
        img_tensor, _, _ = test_transform(image, gt_boxes.copy(), gt_labels.copy())
        inp = np.expand_dims(img_tensor.numpy(), axis=0).astype(np.float32)
        t0 = time.perf_counter()
        _ = compiled_model([inp])
        dt = time.perf_counter() - t0
        if idx >= warmup:
            infer_times.append(dt)

    if not infer_times:
        return 0.0, 0.0
    mean_latency_ms = float(np.mean(infer_times) * 1000.0)
    fps = float(1.0 / np.mean(infer_times))
    return mean_latency_ms, fps


def build_benchmark_inputs(dataset, test_transform, max_samples):
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    inputs = []
    for idx in range(total):
        image, gt_boxes, gt_labels = get_image_and_gt(dataset, idx)
        img_tensor, _, _ = test_transform(image, gt_boxes.copy(), gt_labels.copy())
        inp = np.expand_dims(img_tensor.numpy(), axis=0).astype(np.float32)
        inputs.append(inp)
    return inputs


def benchmark_latency_infer_only(compiled_model, inputs, warmup_samples):
    if not inputs:
        return 0.0, 0.0
    warmup = min(len(inputs), max(0, warmup_samples))
    infer_times = []
    for idx, inp in enumerate(inputs):
        t0 = time.perf_counter()
        _ = compiled_model([inp])
        dt = time.perf_counter() - t0
        if idx >= warmup:
            infer_times.append(dt)
    if not infer_times:
        return 0.0, 0.0
    mean_latency_ms = float(np.mean(infer_times) * 1000.0)
    fps = float(1.0 / np.mean(infer_times))
    return mean_latency_ms, fps


def build_compile_config(perf_hint: str, num_streams: str, num_threads: int):
    cfg = {
        "PERFORMANCE_HINT": perf_hint,
        "NUM_STREAMS": str(num_streams),
    }
    if num_threads and num_threads > 0:
        cfg["INFERENCE_NUM_THREADS"] = str(num_threads)
    return cfg


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
    # Build quantization candidates:
    # - PERFORMANCE: faster, usually larger speedup, may lose more accuracy
    # - MIXED: safer for accuracy
    candidates = []
    if args.quant_mode == "auto":
        candidates = [("performance", nncf.QuantizationPreset.PERFORMANCE), ("mixed", nncf.QuantizationPreset.MIXED)]
    elif args.quant_mode == "performance":
        candidates = [("performance", nncf.QuantizationPreset.PERFORMANCE)]
    else:
        candidates = [("mixed", nncf.QuantizationPreset.MIXED)]

    logging.info("Step4: compile and select INT8 candidate")
    core = ov.Core()
    fp32_cfg = build_compile_config(args.fp32_perf_hint, args.fp32_num_streams, args.fp32_threads)
    int8_cfg = build_compile_config(args.int8_perf_hint, args.int8_num_streams, args.int8_threads)
    logging.info("FP32 compile config: %s", fp32_cfg)
    logging.info("INT8 compile config: %s", int8_cfg)
    compiled_fp32 = core.compile_model(ov_model_fp32, args.device, fp32_cfg)

    best = None
    evaluate_accuracy = (not args.speed_first) or args.strict_accuracy

    if args.speed_first and not args.strict_accuracy:
        logging.info("Speed-first mode enabled: using PERFORMANCE preset without accuracy gating.")
        ov_model_int8 = nncf.quantize(
            ov_model_fp32,
            calibration_dataset=nncf_calib,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=min(len(calib_dataset), args.calib_subset_size),
            fast_bias_correction=True,
        )
        compiled_int8 = core.compile_model(ov_model_int8, args.device, int8_cfg)
        chosen_mode = "performance-speed-first"
        int8_map = -1.0
        map_drop = -1.0
    else:
        fp32_map = evaluate_map(
            compiled_fp32, val_dataset, test_transform, args.val_max_samples,
            args.prob_threshold, args.iou_threshold, args.candidate_size,
        )

        for mode_name, preset in candidates:
            logging.info("Quantizing candidate mode=%s ...", mode_name)
            ov_model_int8_cand = nncf.quantize(
                ov_model_fp32,
                calibration_dataset=nncf_calib,
                preset=preset,
                subset_size=min(len(calib_dataset), args.calib_subset_size),
                fast_bias_correction=True,
            )
            compiled_int8_cand = core.compile_model(ov_model_int8_cand, args.device, int8_cfg)
            int8_map = evaluate_map(
                compiled_int8_cand, val_dataset, test_transform, args.val_max_samples,
                args.prob_threshold, args.iou_threshold, args.candidate_size,
            )
            map_drop = fp32_map - int8_map
            logging.info("[%s] INT8 mAP: %.6f, drop: %.6f", mode_name, int8_map, map_drop)

            if map_drop <= args.max_map_drop:
                best = (mode_name, ov_model_int8_cand, compiled_int8_cand, int8_map, map_drop)
                break

        if best is None:
            msg = (
                f"No INT8 candidate meets max_map_drop={args.max_map_drop:.6f}. "
                "Try increasing calib_subset_size or relaxing max_map_drop."
            )
            if args.strict_accuracy:
                raise RuntimeError(msg)
            logging.warning(msg)
            # fallback to performance when speed_first, else mixed.
            fallback_preset = nncf.QuantizationPreset.PERFORMANCE if args.speed_first else nncf.QuantizationPreset.MIXED
            fallback_name = "performance-fallback" if args.speed_first else "mixed-fallback"
            ov_model_int8 = nncf.quantize(
                ov_model_fp32,
                calibration_dataset=nncf_calib,
                preset=fallback_preset,
                subset_size=min(len(calib_dataset), args.calib_subset_size),
                fast_bias_correction=True,
            )
            compiled_int8 = core.compile_model(ov_model_int8, args.device, int8_cfg)
            int8_map = evaluate_map(
                compiled_int8, val_dataset, test_transform, args.val_max_samples,
                args.prob_threshold, args.iou_threshold, args.candidate_size,
            )
            map_drop = fp32_map - int8_map
            chosen_mode = fallback_name
        else:
            chosen_mode, ov_model_int8, compiled_int8, int8_map, map_drop = best

    if evaluate_accuracy:
        logging.info("FP32 mAP: %.6f", fp32_map)
    logging.info("Chosen INT8 mode: %s", chosen_mode)
    if evaluate_accuracy:
        logging.info("INT8 mAP: %.6f", int8_map)
        logging.info("mAP drop: %.6f", map_drop)
    else:
        logging.info("Accuracy evaluation skipped due to speed-first mode.")

    logging.info("Step5: benchmark FP32 vs INT8 latency/FPS")
    if args.benchmark_infer_only:
        bench_inputs = build_benchmark_inputs(val_dataset, test_transform, args.benchmark_samples)
        fp32_lat_ms, fp32_fps = benchmark_latency_infer_only(compiled_fp32, bench_inputs, args.warmup_samples)
        int8_lat_ms, int8_fps = benchmark_latency_infer_only(compiled_int8, bench_inputs, args.warmup_samples)
    else:
        fp32_lat_ms, fp32_fps = benchmark_latency(
            compiled_fp32, val_dataset, test_transform, args.benchmark_samples, args.warmup_samples
        )
        int8_lat_ms, int8_fps = benchmark_latency(
            compiled_int8, val_dataset, test_transform, args.benchmark_samples, args.warmup_samples
        )
    speedup = (fp32_lat_ms / int8_lat_ms) if int8_lat_ms > 0 else 0.0
    logging.info("FP32 latency: %.3f ms, FPS: %.2f", fp32_lat_ms, fp32_fps)
    logging.info("INT8 latency: %.3f ms, FPS: %.2f", int8_lat_ms, int8_fps)
    logging.info("INT8 speedup: %.3fx", speedup)

    ov.save_model(ov_model_int8, str(int8_xml), compress_to_fp16=False)
    logging.info("FP32 IR: %s", fp32_xml)
    logging.info("INT8 IR: %s", int8_xml)


if __name__ == "__main__":
    main()
