"""
This code is used to convert the pytorch model into an onnx format model.
"""
import argparse
import sys

import torch.onnx

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
parser.add_argument('--input_size', default=720, type=int,
                    help='Input size of the model, must match training size (128/160/320/480/640/720/960/1280)')
parser.add_argument('--model_path', default="models/rfb_720x720/RFB-Epoch-0-mAP-0.0033-ValLoss-1.3267.pth", type=str,
                    help='Path to the PyTorch model file')
parser.add_argument('--net_type', default="RFB", type=str, choices=['slim', 'RFB'],
                    help='Network type (slim or RFB)')
parser.add_argument('--output_path', default="models/onnx/ultra-fast-face-detector-1mb-720.onnx", type=str,
                    help='Path to save the ONNX model file')

args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd

net_type = args.net_type

label_path = "models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'slim':
    model_path = args.model_path
    net = create_mb_tiny_fd(len(class_names), is_test=True)
elif net_type == 'RFB':
    model_path = args.model_path
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True)
else:
    print("unsupport network type.")
    sys.exit(1)
net.load(model_path)
net.eval()
net.to("cuda")

# Set dummy input size based on input_img_size
img_size_dict = {128: [128, 96],
                 160: [160, 120],
                 320: [320, 240],
                 480: [480, 360],
                 640: [640, 480],
                 720: [720, 540],
                 960: [960, 720],
                 1280: [1280, 960]}
h, w = img_size_dict[input_img_size][1], img_size_dict[input_img_size][0]
dummy_input = torch.randn(1, 3, h, w).to("cuda")

torch.onnx.export(net, dummy_input, args.output_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])