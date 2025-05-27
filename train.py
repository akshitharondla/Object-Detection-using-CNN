import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_model
from dataset import VOCDataset
import os
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = VOCDataset(root='./data', transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
model = get_model(num_classes=2)  # 1 class + background
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'fasterrcnn_model.pth')
