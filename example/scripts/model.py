import os
import torch
import requests
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import json
import numpy as np
import torch.optim as optim
from io import BytesIO


# Custom Dataset Class with on-the-fly image downloading
class CustomCocoDataset(Dataset):
    def __init__(self, annotations_file, transforms=None):
        self.transforms = transforms
        with open(annotations_file, 'r') as f:
            self.coco = json.load(f)

        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        self.categories = {category['id']: category['name'] for category in self.coco['categories']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_url = img_info['file_name']  # Here, 'file_name' is actually the URL of the image

        # Download image from URL
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Get annotations for the current image
        annotations = [ann for ann in self.annotations if ann['image_id'] == img_id]

        # Bounding boxes
        boxes = [ann['bbox'] for ann in annotations]

        # Filter out invalid boxes
        valid_boxes = []
        for box in boxes:
            x, y, w, h = box
            if w > 0 and h > 0:
                valid_boxes.append(box)
        boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)

        if len(valid_boxes) == 0:
            # Handle case where no valid boxes are found
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Labels
        labels = [ann['category_id'] for ann in annotations if ann['bbox'] in valid_boxes]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Image ID
        image_id = torch.tensor([img_id])

        # Area of the bounding boxes
        area = torch.as_tensor([ann['area'] for ann in annotations if ann['bbox'] in valid_boxes], dtype=torch.float32)

        # Iscrowd
        iscrowd = torch.as_tensor([ann['iscrowd'] for ann in annotations if ann['bbox'] in valid_boxes],
                                  dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# Custom collate_fn to handle varying-sized images and annotations
def collate_fn(batch):
    return tuple(zip(*batch))


# Function to get pre-trained Faster R-CNN and modify it for custom dataset
def get_model(num_classes):
    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head with a new one that has num_classes categories (including background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Function to train the model
def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_loss = 0
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimize
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    # Ensures multiprocessing is handled correctly on Windows
    torch.multiprocessing.freeze_support()

    # Path to your COCO-style JSON annotations
    annotations_file = 'output_coco.json'

    # Transforms (you can add more augmentations if necessary)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Initialize the dataset
    dataset = CustomCocoDataset(annotations_file, transforms=transform)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Set device (use GPU if available)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Number of classes (including background)
    num_classes = 2  # Modify based on your dataset

    # Get the model
    model = get_model(num_classes)
    model = model.to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    train_model(model, dataloader, optimizer, device, num_epochs=10)
