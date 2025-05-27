import torch
import matplotlib.pyplot as plt
import cv2
from dataset import VOCDataset
from model import get_model
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load('fasterrcnn_model.pth'))
model.to(device)
model.eval()

# Load sample
transform = transforms.Compose([transforms.ToTensor()])
dataset = VOCDataset(root='./data', transform=transform)
img, _ = dataset[0]
img_tensor = img.to(device)

with torch.no_grad():
    prediction = model([img_tensor])

# Visualize
img_np = img.permute(1, 2, 0).cpu().numpy().copy()
for box in prediction[0]['boxes'].cpu():
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(img_np)
plt.title("Predictions")
plt.axis("off")
plt.show()
