import torch
import numpy as np
from torch import nn, optim
from tactile.model import CustomModel
from torch.utils.data import DataLoader, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print('Device: ', device)

# 데이터
image_data = np.load('./data_make/image.npy')
label_data = np.load('./data_make/label.npy')

image_data = image_data.astype(np.float32)
print("Data shape: ", image_data.shape)

image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
image_data = image_data.reshape(-1, 1, 25, 25)
print("Complete Normalization")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.label[idx]
        return image, label

dataset = CustomDataset(image_data, label_data)
train_size = int(0.8 * len(dataset))
validaion_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - validaion_size

train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, validaion_size, test_size])

print("Train set: ", len(train_dataset))
print("Validation set: ", len(valid_dataset))
print("Test set: ", len(test_dataset))

model = CustomModel(3).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=True)

train_losses = []
valid_losses = []
train_accurs = []
valid_accurs = []

epochs = 100
best_loss = 10 ** 9
patience_limit = 5
patience_check = 0
for epoch in range(epochs):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            valid_total += labels.size(0)
            valid_correct += predicted.eq(labels).sum().item()

    if valid_loss < best_loss:
        best_loss = valid_loss
        patience_check = 0
    else:
        patience_check += 1
        if patience_check > patience_limit:
            print("Early Stopping")
            break
    train_accuracy = 100 * train_correct / train_total
    valid_accuracy = 100 * valid_correct / valid_total
    train_loss_avg = train_loss / len(train_loader)
    valid_loss_avg = valid_loss / len(valid_loader)

    train_losses.append(train_loss_avg)
    valid_losses.append(valid_loss_avg)
    train_accurs.append(train_accuracy)
    valid_accurs.append(valid_accuracy)

    print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss_avg:.5f} | Validation Loss: {valid_loss_avg:.5f} | Train Accuracy: {train_accuracy:.2f}% | Validation Accuracy: {valid_accuracy:.2f}%')

print("Training Complete")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accurs, label='Train Accuracy')
plt.plot(valid_accurs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./data_make/loss_acc.png')

# 모델 저장
torch.save(model.state_dict(), './data_make/model.pth')

predicted_label = []
actual_label = []
model.eval()

for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, predicted = outputs.max(1)

    predicted_label.extend(predicted.tolist())
    actual_label.extend(labels.tolist())

cm = confusion_matrix(actual_label, predicted_label)
labels_class = ['CYH', 'KSH', 'SCH']

plt.cla()
plt.clf()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')

plt.savefig('./data_make/confusion_matrix.png')