"""
Fabric Defect Classifier - Training Script
============================================
Run this in Google Colab (free GPU: Runtime > Change runtime type > T4 GPU)

BEFORE RUNNING:
1. Zip your dataset folder (the one with 9 class subfolders) and upload it to
   Colab, OR upload it to Google Drive and mount Drive.
2. Update DATA_DIR below to point at the folder containing the 9 class folders.

Expected structure:
    DATA_DIR/
        defect-free/*.jpg
        hole/*.jpg
        horizontal/*.jpg
        vertical/*.jpg
        lines/*.jpg
        pinched-fabric/*.jpg
        needle-mark/*.jpg
        broken-stitch/*.jpg
        stain/*.jpg
"""

# ── Setup ──────────────────────────────────────────────────────────────────
# !pip install torch torchvision grad-cam scikit-learn matplotlib -q

import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/content/fabric_dataset"   # <-- update this path
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data loading & augmentation ──────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")
print(f"Total images: {len(full_dataset)}")

# 70/15/15 train/val/test split
n = len(full_dataset)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
# Val/test should use eval_transform (no augmentation) -- swap the transform
val_ds.dataset.transform = eval_transform
test_ds.dataset.transform = eval_transform

# ── Handle class imbalance with weighted sampling ────────────────────────
targets = [full_dataset.samples[i][1] for i in train_ds.indices]
class_counts = np.bincount(targets, minlength=num_classes)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = [class_weights[t] for t in targets]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Class distribution in training set: {dict(zip(class_names, class_counts))}")

# ── Model: transfer learning on ResNet50 ─────────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone initially

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

# ── Training loop ─────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

best_val_acc = 0.0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # Unfreeze backbone after a few epochs for fine-tuning
    if epoch == 5:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
        print("Unfroze backbone, continuing fine-tuning at lower LR")

    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "fabric_defect_classifier.pt")
print(f"\nBest validation accuracy: {best_val_acc:.4f}")
print("Saved model weights to fabric_defect_classifier.pt")

# ── Final evaluation on held-out test set ────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n=== Test Set Performance (report these honest numbers on your CV) ===")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion_matrix.png -- put this in your README/portfolio")

# ── Grad-CAM: visualize what the model is looking at ────────────────────
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def generate_gradcam(image_path, model, class_names, save_path="gradcam_output.png"):
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img) / 255.0
    input_tensor = eval_transform(img).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title(f"Predicted: {class_names[pred_idx]} ({confidence:.1%})")
    plt.axis("off")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")
    return class_names[pred_idx], confidence

# Example usage after training:
# generate_gradcam("/content/fabric_dataset/hole/some_test_image.jpg", model, class_names)


# ── Show Grad-CAM on a handful of random images from the dataset ────────
import random
import glob

def show_gradcam_samples(model, class_names, dataset_dir, num_samples=6):
    """
    Picks a few random images across your dataset folders and displays
    the Grad-CAM heatmap for each one, side by side. This is your
    "does the model actually look at the right spot" sanity check --
    and your best screenshot for the portfolio/demo.
    """
    # Step 1: collect every image path from every class folder
    all_images = []
    for class_name in class_names:
        class_folder = os.path.join(dataset_dir, class_name)
        images_in_class = (
            glob.glob(os.path.join(class_folder, "*.jpg"))
            + glob.glob(os.path.join(class_folder, "*.jpeg"))
            + glob.glob(os.path.join(class_folder, "*.png"))
        )
        all_images.extend(images_in_class)

    # Step 2: pick a random handful of them
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))

    # Step 3: set up Grad-CAM once (reused for every image, faster than
    # recreating it inside the loop)
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Step 4: run each image through the model + Grad-CAM, plot side by side
    fig, axes = plt.subplots(1, len(sample_images), figsize=(4 * len(sample_images), 4))
    if len(sample_images) == 1:
        axes = [axes]  # keep it a list even when there's only 1 image

    for ax, image_path in zip(axes, sample_images):
        img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img) / 255.0
        input_tensor = eval_transform(img).unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

        true_label = os.path.basename(os.path.dirname(image_path))
        ax.imshow(visualization)
        ax.set_title(f"true: {true_label}\npred: {class_names[pred_idx]} ({confidence:.1%})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_samples_grid.png", dpi=150)
    plt.show()
    print("Saved gradcam_samples_grid.png")

# Example usage after training:
# show_gradcam_samples(model, class_names, DATA_DIR, num_samples=6)
