import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Custom dataset class
class GurmukhiDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def load_images_from_folder(base_folder):
    X_data = []
    y_data = []

    for digit in range(10):
        folder_path = os.path.join(base_folder, str(digit))
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X_data.append(img.flatten())
                y_data.append(digit)

    return np.array(X_data), np.array(y_data)


# Load the data
X_train_data, y_train_data = load_images_from_folder("train")
X_val_data, y_val_data = load_images_from_folder("val")

# Normalize the data
X_train_data = X_train_data / 255.0
X_val_data = X_val_data / 255.0

# Create dataset and dataloader
train_dataset = GurmukhiDataset(X_train_data, y_train_data)
val_dataset = GurmukhiDataset(X_val_data, y_val_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Define the neural network architecture

class gurmukhi_digit_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size[2], output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Initialize the model, loss, and optimizer
input_size = X_train_data.shape[1]
hidden_size = [128, 64, 32]
output_size = 10
dropout_p = 0.4  # Increased dropout probability

model = gurmukhi_digit_classifier(input_size, hidden_size, output_size, dropout_p)

# # Add weight decay to the optimizer for L2 regularization (increased value)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=9e-4)

# Train the neural network
epochs = 100
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images = images
        labels = labels

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()

    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images
            labels = labels

            output = model(images)
            loss = loss_fn(output, labels)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')




def preprocess_image_unseeen(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (32, 32))
    image_array = np.array(resized_image).flatten()
    image_array = image_array / 255.0
    return image_array

def predict(image_path, model):
    model.eval()
    image_array = preprocess_image_unseeen(image_path)
    input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    output = model(input_tensor)
    _, predicted_digit = torch.max(output, 1)
    return predicted_digit.item()

# Example usage
image_path = './val/8/1.tiff'
predicted_digit = predict(image_path, model)
print(f"Predicted digit: {predicted_digit}")