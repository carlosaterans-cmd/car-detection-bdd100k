import os
import json
import time
from pprint import pprint
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import CarDetectionDataset, get_transform, get_car_bbox, collate_fn
from train import get_model

def validate_model(model, val_img_dir, val_json_dir, device, num_images=400):
    """Calculates mAP metrics on validation data."""
    val_dataset = CarDetectionDataset(img_dir=val_img_dir, json_dir=val_json_dir, N_img=num_images, transform=get_transform())
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model.eval()
    metric = MeanAveragePrecision()
    
    print("Starting validation...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            res_targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            
            metric.update(outputs, res_targets)

    results = metric.compute()
    pprint(results)
    return results

def save_test_results(model, test_img_dir, test_json_dir, output_dir, device, threshold=0.5, num_images=10):
    """Runs predictions on test data and saves visual results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    all_img_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_files = np.random.choice(all_img_files, num_images, replace=False)
    
    print(f"Processing {len(img_files)} test images...")

    for img_name in img_files:
        img_path = os.path.join(test_img_dir, img_name)
        json_path = os.path.join(test_json_dir, os.path.splitext(img_name)[0] + '.json')
        
        orig_img = Image.open(img_path).convert("RGB")
        width, height = orig_img.size
        
        input_img = orig_img.resize((640, 640))
        img_tensor = F.to_tensor(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(img_tensor)

        img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
        
        # Draw Ground Truth
        with open(json_path, 'r') as f:
            scene = json.load(f)
        for box in get_car_bbox(scene["frames"][0]["objects"]):
            cv2.rectangle(img_cv, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (0, 0, 255), 2)

        # Draw Predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        for i, score in enumerate(scores):
            if score > threshold:
                xmin = int(boxes[i][0] * (width / 640))
                xmax = int(boxes[i][2] * (width / 640))
                ymin = int(boxes[i][1] * (height / 640))
                ymax = int(boxes[i][3] * (height / 640))

                cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img_cv, f"Car: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(output_dir, f"res_{img_name}"), img_cv)
    print(f"Visual results saved to {output_dir}")