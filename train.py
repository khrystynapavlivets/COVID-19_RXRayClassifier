import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from DataPred import load_covid_dataset, display_samples
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random

# %%

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device in use:", device)

# %%
df, classes = load_covid_dataset()

# Display 3 samples from each class
display_samples(df, classes, samples_per_class=3)

X = df["image_path"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# %%
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


# %%
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


# %%
train_dataset = CovidDataset(X_train, y_train_enc, transform=transform)
test_dataset = CovidDataset(X_test, y_test_enc, transform=transform)
# %%
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# %%
img, label = train_dataset[0]
print(img.shape)
print(label)

# %%
# %%
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
NUM_CLASSES = len(CLASSES)


# set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # out: (batch_size, 1, 224, 224) - 1 channel (grayscale)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (BS, 32, 224, 224)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # (BS, 32, 124, 124)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (BS, 64, 124, 124)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # (BS, 64, 56, 56)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (BS, 128, 56, 56)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # (BS, 128, 28, 28)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (BS, 256, 28, 28)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # (BS, 256, 14, 14)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # (BS, 256, 1, 1)ʼ

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
model = ImageMulticlassClassificationNet()
model.to(device)

# %%
LR = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %%
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # zero gradients

        outputs = model(inputs)  # forward pass

        loss = loss_fn(outputs, labels)  # calculate loss

        loss.backward()  # backward pass

        optimizer.step()  # update weights

        total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

# %% test
model.to(device)
model.eval()  # Putting the model into evaluation mode
y_test = []
y_test_hat = []

for i, data in enumerate(test_dataloader, 0):
    inputs, y_test_temp = data
    inputs = inputs.to(device)
    y_test_temp = y_test_temp.to(device)
    with torch.no_grad():
        y_test_hat_temp = model(inputs)
        _, predicted = torch.max(y_test_hat_temp, 1)

    y_test.extend(y_test_temp.cpu().numpy())
    y_test_hat.extend(predicted.cpu().numpy())


# %%
acc = accuracy_score(y_test, y_test_hat)
print(f"Accuracy: {acc*100:.2f} %")

# %% confusion matrix
cm = confusion_matrix(y_test, y_test_hat)
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
# %%
# %%
# Display some test images with true and predicted labels
for i in random.sample(range(len(test_dataset)), 5):
    image, label = test_dataset[i]  # image: тензор (1, H, W), label: int
    pred_label = np.argmax(y_test_hat[i])  # припускаю, що y_test_hat[i] існує

    img_np = image.squeeze().numpy()
    img_np = img_np * 0.5 + 0.5  # денормалізація
plt.figure(figsize=(2, 2))
plt.imshow(img_np, cmap="gray")
plt.title(f"True: {CLASSES[label]}, Pred: {CLASSES[pred_label]}")
plt.axis("off")
plt.show()


# %%

for i in random.sample(range(len(test_dataset)), 5):

    image, true_label, image_path = test_dataset[i]

    image = Image.open(image_path).convert("L")

    image_tensor = transform(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_label_index = output.argmax(dim=1).item()

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"True: {CLASSES[true_label]}, Pred: {CLASSES[pred_label_index]}")
    plt.axis("off")
    plt.show()
# %%
