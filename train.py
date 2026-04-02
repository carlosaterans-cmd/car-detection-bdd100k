import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

from dataset import CarDetectionDataset, get_transform, collate_fn

def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize dataset & dataloader
    train_dataset = CarDetectionDataset(
        img_dir=args.img_dir, 
        json_dir=args.json_dir,
        N_img=args.num_images,
        transform=get_transform()
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    start = time.time()

    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train() 
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad() 
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {losses.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}\n")
        
        torch.save(model.state_dict(), 'car_detector_last.pth')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'car_detector_best.pth')
            print("New best model saved.")

    print(f"Training Complete! Total time: {(time.time() - start)/3600:.3f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Car Detection Model")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to BDD100k training images")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to BDD100k training labels")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to use")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    
    args = parser.parse_args()
    main(args)