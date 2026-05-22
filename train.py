# %%  Imports and Configuration
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models
from DataPred import load_covid_dataset, display_samples
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device in use:", device)


# %% Data Loading and Preparation
df, classes = load_covid_dataset()

# Display 3 samples from each class
display_samples(df, classes, samples_per_class=3)

X = df["image_path"].values
y = df["label"].values

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Encode categorical labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Calculate class weights to handle dataset imbalance
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train_enc), y=y_train_enc
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


# %% Transforms and Dataset Definition
# Data augmentation for training
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # ResNet18 expects 3 channels
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Standard preprocessing for testing
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CovidDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long), img_path


# Initialize Datasets and Loaders
train_dataset = CovidDataset(X_train, y_train_enc, transform=train_transform)
test_dataset = CovidDataset(X_test, y_test_enc, transform=test_transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

img, label, img_path = train_dataset[0]
print(img.shape)
print(label)
print(img_path)



# %% Model Architecture
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
NUM_CLASSES = len(CLASSES)

# Model architecture using Transfer Learning (Pre-trained ResNet18)
class CovidResNet(nn.Module):
    def __init__(self, num_classes):
        super(CovidResNet, self).__init__()
        # Load pre-trained ResNet18 weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace the final fully connected layer to match our number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),  # Regularization to prevent overfitting
            nn.Linear(num_ftrs, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)

# Initialize model and move to device
model = CovidResNet(NUM_CLASSES)
model.to(device)



# %% Training Setup and Loop
# Training configuration
LR = 0.001
loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss for imbalance
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# LR Scheduler to reduce learning rate when progress stalls
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=2, factor=0.5
)

# Main training loop
NUM_EPOCHS = 7

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_dataloader, 0):
        inputs, labels, img_paths = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            current_acc = 100 * correct / total
            print(
                f"  Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%"
            )

    # Print epoch summary
    avg_loss = total_loss / len(train_dataloader)
    train_acc = 100 * correct / total
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%"
    )

    # Update learning rate based on epoch loss
    scheduler.step(avg_loss)



# %% Evaluation and Metrics
# Evaluate the model on the test set
model.to(device)
model.eval()
y_test_true = []
y_test_hat = []

for i, data in enumerate(test_dataloader, 0):
    inputs, labels, _ = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    y_test_true.extend(labels.cpu().numpy())
    y_test_hat.extend(predicted.cpu().numpy())


# Print final test accuracy
acc = accuracy_score(y_test_true, y_test_hat)
print(f"Final Test Accuracy: {acc*100:.2f} %")

# Generate and display the Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_hat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# %%  Visualization of Results
# Visualization: Display several random images with results
for i in random.sample(range(len(test_dataset)), 5):
    image, label, _ = test_dataset[i]  # image shape: (3, 224, 224)
    pred_label = y_test_hat[i]

    # Inverse normalization using ImageNet parameters
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    img_denorm = inv_normalize(image)

    # Convert from (C, H, W) to (H, W, C) for matplotlib display
    img_np = img_denorm.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.title(f"True: {CLASSES[label]}, Pred: {CLASSES[pred_label]}")
    plt.axis("off")
    plt.show()
