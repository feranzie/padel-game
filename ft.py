import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import SGD

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get the transformation function
def get_transform():
    return ToTensor()

# Load datasets
train_dataset = CocoDetection(root='/notebooks/finetune/Phase-I-2/train', 
                              annFile='/notebooks/finetune/Phase-I-2/train/_annotations.coco.json', 
                              transform=get_transform())

val_dataset = CocoDetection(root='/notebooks/finetune/Phase-I-2/valid', 
                            annFile='/notebooks/finetune/Phase-I-2/valid/_annotations.coco.json', 
                            transform=get_transform())

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, 
                          collate_fn=lambda x: tuple(zip(*x)))

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, 
                        collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Faster R-CNN model with ResNet50 backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Freeze the backbone (ResNet50) parameters
for param in model.backbone.parameters():
    param.requires_grad = False

# Modify the model to fit your custom number of classes
num_classes = 91  # Adjust based on your dataset (1 class + background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer (only for trainable parameters) and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Function to convert COCO-style annotations to the format expected by Faster R-CNN
def coco_target_to_fastrcnn_target(coco_target):
    boxes = []
    labels = []

    for obj in coco_target:
        x_min, y_min, width, height = obj['bbox']

        # Ensure width and height are positive
        if width > 0 and height > 0:
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(obj['category_id'])

    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
    }

# Function to evaluate the model and compute validation loss
def evaluate_model(model, val_loader, device):
    model.train()  # Keep in training mode to calculate loss
    total_loss = 0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, targets in val_loader:
            images = [image.to(device) for image in images]

            # Convert COCO targets to format expected by Faster R-CNN
            targets = [coco_target_to_fastrcnn_target(t) for t in targets]

            # Skip images without valid bounding boxes
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t['boxes'].shape[0] > 0]
            images = [img for img, t in zip(images, targets) if t['boxes'].shape[0] > 0]

            if len(images) == 0:
                continue

            # Forward pass to get the loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

# Early stopping function
def early_stopping(validation_losses, patience):
    if len(validation_losses) > patience:
        # Check if the last `patience` losses have not decreased
        if all(validation_losses[i] >= validation_losses[i-1] for i in range(-patience, 0)):
            return True
    return False

# Training loop with validation and early stopping
num_epochs = 30
patience = 3  # Early stopping patience
best_val_loss = float('inf')
validation_losses = []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0

    for images, targets in train_loader:
        images = [image.to(device) for image in images]

        # Convert COCO targets to format expected by Faster R-CNN
        targets = [coco_target_to_fastrcnn_target(t) for t in targets]

        # Skip images without valid bounding boxes
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets if t['boxes'].shape[0] > 0]
        images = [img for img, t in zip(images, targets) if t['boxes'].shape[0] > 0]

        if len(images) == 0:
            continue

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}")

    # Evaluate after each epoch
    val_loss = evaluate_model(model, val_loader, device)
    validation_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_fasterrcnn_resnet50_tennis_ball.pth')  # Save the best model

    # Early stopping
    if early_stopping(validation_losses, patience):
        print(f"Early stopping at epoch {epoch+1}")
        break

# After training, save the final model
torch.save(model.state_dict(), 'final_fasterrcnn_resnet50_tennis_ball.pth')
