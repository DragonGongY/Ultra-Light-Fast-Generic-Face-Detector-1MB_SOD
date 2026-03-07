"""
This code is the main training code.
"""
import argparse
import itertools
import logging
import os
import sys
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.yolo_dataset import YOLODataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import fd_config
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels

parser = argparse.ArgumentParser(
    description='train With Pytorch')

parser.add_argument("--dataset_type", default="yolo", type=str,
                    help='Specify dataset type. Currently support voc, yolo.')

parser.add_argument('--yolo_data_yaml', default="/home/share_4T/workspace/algorithm/det/YOLO/data_20260209_001613/data.yaml", type=str,
                    help='Path to YOLO data.yaml configuration file')

parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--net', default="RFB",
                    help="The network architecture ,optional(RFB , slim)")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=100, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--val_batch_size', default=100, type=int,
                    help='Batch size for validation (default: same as training batch_size)')
parser.add_argument('--prefetch_factor', default=4, type=int,
                    help='Number of batches to prefetch for dataloader')
parser.add_argument('--validation_epochs', default=1, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--log_dir', default='./models/Ultra-Light(1MB)_&_Fast_Face_Detector/logs',
                    help='lod dir')
parser.add_argument('--cuda_index', default="0", type=str,
                    help='Choose cuda index.If you have 4 GPUs, you can set it like 0,1,2,3')
parser.add_argument('--power', default=2, type=int,
                    help='poly lr pow')
parser.add_argument('--overlap_threshold', default=0.35, type=float,
                    help='overlap_threshold')
parser.add_argument('--optimizer_type', default="Adam", type=str,
                    help='optimizer_type,optional(SGD,Adam)')
parser.add_argument('--input_size', default=960, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/720/960/1280')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

def load_yolo_data_yaml(yaml_path):
    """Load YOLO data.yaml configuration file"""
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

input_img_size = args.input_size  # define input size ,default optional(128/160/320/480/640/1280)
logging.info("inpu size :{}".format(input_img_size))
define_img_size(input_img_size)  # must put define_img_size() before 'import fd_config'

# Set device based on cuda_index argument
if args.use_cuda and torch.cuda.is_available():
    cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
    DEVICE = torch.device(f"cuda:{cuda_index_list[0]}")
    torch.backends.cudnn.benchmark = True
    logging.info(f"Use Cuda: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    logging.info("Use CPU")

def bbox_overlaps(boxes1, boxes2):
    """
    Calculate the IoU between two sets of boxes
    Args:
        boxes1: (N, 4) tensor or numpy array, each box is [x1, y1, x2, y2]
        boxes2: (M, 4) tensor or numpy array, each box is [x1, y1, x2, y2]
    Returns:
        overlaps: (N, M) tensor or numpy array, IoU of each pair of boxes
    """
    # Convert to numpy arrays if they are tensors
    if hasattr(boxes1, 'numpy'):
        boxes1 = boxes1.numpy()
    if hasattr(boxes2, 'numpy'):
        boxes2 = boxes2.numpy()
    
    # Calculate area of each box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate intersection
    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
    
    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union area
    union = area1[:, None] + area2 - intersection
    
    # Calculate IoU with numerical stability
    overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    
    return overlaps

def lr_poly(base_lr, iter):
    return base_lr * ((1 - float(iter) / args.num_epochs) ** (args.power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.lr, i_iter)
    optimizer.param_groups[0]['lr'] = lr


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    
    # Use tqdm progress bar
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    
    for i, data in pbar:
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        
        # Update progress bar postfix
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Reg': f'{avg_reg_loss:.4f}',
                'Clf': f'{avg_clf_loss:.4f}'
            })

            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
    
    pbar.close()


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    
    # Use tqdm progress bar for validation
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Validation (Loss)")
    
    for i, data in pbar:
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss / num:.4f}',
            'Reg': f'{running_regression_loss / num:.4f}',
            'Clf': f'{running_classification_loss / num:.4f}'
        })
    
    pbar.close()
    
    # Synchronize to ensure GPU operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


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
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]

    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def evaluate_detection(loader, net, device, iou_thresh=0.5, top_k=100):

    net.eval()

    if hasattr(net, "module"):
        num_classes = net.module.num_classes
    else:
        num_classes = net.num_classes

    predictions = []
    gts = []

    with torch.no_grad():

        pbar = tqdm(loader, desc="Validation mAP")

        for images, boxes, labels in pbar:

            images = images.to(device)

            confidences, locations = net(images)

            batch = images.shape[0]

            for b in range(batch):

                conf = confidences[b]
                loc = locations[b]

                scores, cls = torch.max(conf, dim=1)

                mask = (cls > 0) & (scores > 0.01)

                scores = scores[mask]
                cls = cls[mask]
                loc = loc[mask]

                if scores.shape[0] > 0:

                    scores, order = scores.sort(descending=True)
                    order = order[:top_k]

                    pred_boxes = loc[order].cpu().numpy()
                    pred_labels = cls[order].cpu().numpy()
                    pred_scores = scores[:top_k].cpu().numpy()

                else:

                    pred_boxes = np.zeros((0,4))
                    pred_labels = np.zeros(0)
                    pred_scores = np.zeros(0)

                predictions.append((pred_boxes, pred_labels, pred_scores))

                gt_box = boxes[b].cpu().numpy()
                gt_label = labels[b].cpu().numpy()

                valid = gt_label > 0

                gts.append((gt_box[valid], gt_label[valid]))

    aps = []

    for cls_id in range(1, num_classes):

        tp = []
        fp = []
        scores = []
        total_gt = 0

        for i in range(len(predictions)):

            pred_box, pred_label, pred_score = predictions[i]
            gt_box, gt_label = gts[i]

            pred_mask = pred_label == cls_id
            gt_mask = gt_label == cls_id

            pred_box = pred_box[pred_mask]
            pred_score = pred_score[pred_mask]

            gt_box = gt_box[gt_mask]

            total_gt += len(gt_box)

            if len(pred_box) == 0:
                continue

            scores.extend(pred_score)

            if len(gt_box) == 0:

                fp.extend([1]*len(pred_box))
                tp.extend([0]*len(pred_box))
                continue

            ious = bbox_iou(pred_box, gt_box)

            matched = set()

            for j in range(len(pred_box)):

                iou = np.max(ious[j])
                idx = np.argmax(ious[j])

                if iou >= iou_thresh and idx not in matched:

                    tp.append(1)
                    fp.append(0)
                    matched.add(idx)

                else:

                    tp.append(0)
                    fp.append(1)

        if total_gt == 0:
            continue

        if len(tp) == 0:
            aps.append(0)
            continue

        scores = np.array(scores)
        tp = np.array(tp)
        fp = np.array(fp)

        order = np.argsort(-scores)

        tp = tp[order]
        fp = fp[order]

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / (total_gt + 1e-6)
        precision = tp / (tp + fp + 1e-6)

        aps.append(compute_ap(recall, precision))

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps))


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    if args.net == 'slim':
        create_net = create_mb_tiny_fd
        config = fd_config
    elif args.net == 'RFB':
        create_net = create_Mb_Tiny_RFB_fd
        config = fd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, args.overlap_threshold)

    test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    logging.info("Prepare training datasets.")
    datasets = []
    
    if args.dataset_type == 'yolo':
        if not args.yolo_data_yaml:
            raise ValueError("For YOLO dataset type, --yolo_data_yaml must be specified")
        
        yolo_data_config = load_yolo_data_yaml(args.yolo_data_yaml)
        dataset_root = yolo_data_config.get('path')
        
        logging.info(f"Loaded YOLO data config from {args.yolo_data_yaml}")
        logging.info(f"Dataset root: {dataset_root}")
        logging.info(f"Train path: {yolo_data_config.get('train')}")
        logging.info(f"Val path: {yolo_data_config.get('val')}")
        logging.info(f"Classes: {yolo_data_config.get('names')}")
        
        train_dataset = YOLODataset(dataset_root, transform=train_transform,
                                   target_transform=target_transform,
                                   split='train',
                                   data_config=yolo_data_config)
        label_file = os.path.join(args.checkpoint_folder, "yolo-model-labels.txt")
        store_labels(label_file, train_dataset.class_names)
        num_classes = len(train_dataset.class_names)
        
        val_dataset = YOLODataset(dataset_root, transform=test_transform,
                                   target_transform=target_transform, is_test=True,
                                   split='val',
                                   data_config=yolo_data_config)
        
        logging.info(f"Stored labels into file {label_file}.")
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                              persistent_workers=True if args.num_workers > 0 else False)
        logging.info("Validation dataset size: {}".format(len(val_dataset)))
        val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size
        val_loader = DataLoader(val_dataset, val_batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                            persistent_workers=True if args.num_workers > 0 else False)
    
    elif args.dataset_type == 'voc':
        if not args.datasets or not args.validation_dataset:
            raise ValueError("For VOC dataset type, --datasets and --validation_dataset must be specified")
        
        for dataset_path in args.datasets:
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
            datasets.append(dataset)
        
        logging.info(f"Stored labels into file {label_file}.")
        train_dataset = ConcatDataset(datasets)
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                              persistent_workers=True if args.num_workers > 0 else False)
        
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
        logging.info("validation dataset size: {}".format(len(val_dataset)))
        val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size
        val_loader = DataLoader(val_dataset, val_batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                            persistent_workers=True if args.num_workers > 0 else False)
    
    else:
        raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
    logging.info("Build network.")
    net = create_net(num_classes)

    # add multigpu_train
    cuda_index_list = None
    if torch.cuda.device_count() >= 1:
        cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        net = nn.DataParallel(net, device_ids=cuda_index_list)
        logging.info("use gpu :{}".format(cuda_index_list))

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        if cuda_index_list:
            params = [
                {'params': net.module.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    net.module.source_layer_add_ons.parameters(),
                    net.module.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.module.regression_headers.parameters(),
                    net.module.classification_headers.parameters()
                )}
            ]
        else:
            params = [
                {'params': net.base_net.parameters(), 'lr': base_net_lr},
                {'params': itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ), 'lr': extra_layers_lr},
                {'params': itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters()
                )}
            ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
        logging.info("use Adam optimizer")
    else:
        logging.fatal(f"Unsupported optimizer: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")
    if args.optimizer_type != "Adam":
        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,
                                    gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'cosine':
            logging.info("Uses CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
        elif args.scheduler == 'poly':
            logging.info("Uses PolyLR scheduler.")
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.optimizer_type != "Adam":
            if args.scheduler != "poly":
                if epoch != 0:
                    scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        if args.scheduler == "poly":
            adjust_learning_rate(optimizer, epoch)
        logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            logging.info("lr rate :{}".format(optimizer.param_groups[0]['lr']))
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            
            # Compute mAP
            val_map = evaluate_detection(val_loader, net, DEVICE)
            
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}, " +
                f"Validation mAP: {val_map:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-mAP-{val_map:.4f}-ValLoss-{val_loss:.4f}.pth")
            if cuda_index_list:
                net.module.save(model_path)
            else:
                net.save(model_path)
            logging.info(f"Saved model {model_path}")
