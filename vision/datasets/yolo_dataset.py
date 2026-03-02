import logging
import os
import pathlib

import cv2
import numpy as np


class YOLODataset:
    def __init__(self, root, transform=None, target_transform=None, is_test=False, 
                 keep_difficult=False, split='train', data_config=None):
        """Dataset for YOLO format data.
        Args:
            root: the root of the YOLO dataset
            split: dataset split ('train' or 'val')
            data_config: YOLO data.yaml configuration dict
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.is_test = is_test
        self.split = split
        self.data_config = data_config

        if data_config:
            train_path = data_config.get('train', 'images/train')
            val_path = data_config.get('val', 'images/val')
            
            if split == 'train':
                self.images_dir = self.root / train_path
            else:
                self.images_dir = self.root / val_path
            
            self.labels_dir = self.images_dir.parent.parent / 'labels' / split
            
            if not self.images_dir.exists():
                self.images_dir = self.root / 'images' / split
                self.labels_dir = self.root / 'labels' / split
        else:
            self.images_dir = self.root / "images" / split
            self.labels_dir = self.root / "labels" / split

        if not self.images_dir.exists():
            raise ValueError(f"Cannot find images directory: {self.images_dir}")

        if not self.labels_dir.exists():
            raise ValueError(f"Cannot find labels directory: {self.labels_dir}")

        self.ids = self._read_image_ids()

        if data_config and 'names' in data_config:
            self._load_classes_from_config(data_config['names'])
        else:
            label_file_name = self.root / "classes.txt"
            if os.path.isfile(label_file_name):
                self._load_classes_from_file(label_file_name)
            else:
                logging.info("No classes file, using default classes.")
                self.class_names = ('BACKGROUND', 'face')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        logging.info(f"YOLO Dataset loaded with {len(self.ids)} images and {len(self.class_names)} classes")
        logging.info(f"Split: {split}, Images dir: {self.images_dir}, Labels dir: {self.labels_dir}")
        logging.info(f"Classes: {self.class_names}")

    def _load_classes_from_config(self, names_dict):
        if isinstance(names_dict, dict):
            classes = [names_dict[i] for i in sorted(names_dict.keys())]
        elif isinstance(names_dict, list):
            classes = names_dict
        else:
            classes = []
        classes.insert(0, 'BACKGROUND')
        classes = [elem.replace(" ", "") for elem in classes]
        self.class_names = tuple(classes)
        logging.info(f"YOLO Classes loaded from data.yaml: {self.class_names}")

    def _load_classes_from_file(self, classes_file):
        classes = []
        with open(classes_file, 'r') as f:
            for line in f:
                class_name = line.strip()
                if class_name:
                    classes.append(class_name)
        classes.insert(0, 'BACKGROUND')
        classes = [elem.replace(" ", "") for elem in classes]
        self.class_names = tuple(classes)
        logging.info(f"YOLO Classes read from file: {self.class_names}")

    def _read_image_ids(self):
        ids = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in image_extensions:
            for image_file in self.images_dir.glob(f'*{ext}'):
                ids.append(image_file.stem)
        
        for ext in image_extensions:
            for image_file in self.images_dir.glob(f'*{ext.upper()}'):
                if image_file.stem not in ids:
                    ids.append(image_file.stem)
        
        return sorted(ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        boxes, labels = self._get_annotation(image_id, image)
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id, image=None):
        label_file = self.labels_dir / f"{image_id}.txt"
        
        if not label_file.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        boxes = []
        labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    if image is None:
                        image = self._read_image(image_id)
                    img_height, img_width = image.shape[:2]
                    
                    x1 = (x_center - width / 2) * img_width
                    y1 = (y_center - height / 2) * img_height
                    x2 = (x_center + width / 2) * img_width
                    y2 = (y_center + height / 2) * img_height
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)
        
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _read_image(self, image_id):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in image_extensions:
            image_file = self.images_dir / f"{image_id}{ext}"
            if image_file.exists():
                image = cv2.imread(str(image_file))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image
        
        for ext in image_extensions:
            image_file = self.images_dir / f"{image_id}{ext.upper()}"
            if image_file.exists():
                image = cv2.imread(str(image_file))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image
        
        raise FileNotFoundError(f"Image file not found for {image_id}")
