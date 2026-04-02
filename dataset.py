import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def get_car_bbox(objects):
    """Extracts car bounding boxes from BDD100K JSON format."""
    bboxes = []
    for obj in objects:
        if obj['category'] == 'car':
            x1 = obj['box2d']['x1']
            x2 = obj['box2d']['x2']
            y1 = obj['box2d']['y1']
            y2 = obj['box2d']['y2']
            bboxes.append([x1, x2, y1, y2])
    return bboxes

class CarDetectionDataset(Dataset):
    """Custom Dataset for loading car images and bounding box annotations."""
    def __init__(self, img_dir, json_dir, N_img, transform=None, target_size=(640, 640)):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.N_img = N_img
        self.target_size = target_size
        self.transform = transform
        
        # Map all image files in the directory
        self.all_img_files = sorted([f for f in os.listdir(img_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Select subset of images
        self.img_files = np.random.choice(self.all_img_files, self.N_img, replace=False)
        self.class_to_id = {'car': 1}
        print(f"Found {len(self.all_img_files)} images, using {len(self.img_files)}.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(self.json_dir, json_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            raise
            
        orig_w, orig_h = image.size
        image = image.resize(self.target_size)
        
        ratio_w = self.target_size[0] / orig_w
        ratio_h = self.target_size[1] / orig_h
        
        with open(json_path, 'r') as f:
            scene = json.load(f)

        boxes = []
        labels = []
        objects = scene["frames"][0]["objects"]
        car_bboxes = get_car_bbox(objects)
        
        for box in car_bboxes:
            x_min, x_max, y_min, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Scale coordinates
            x_min *= ratio_w
            x_max *= ratio_w
            y_min *= ratio_h
            y_max *= ratio_h
            
            if width > 0 and height > 0:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.class_to_id.get('car', 1))
        
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))