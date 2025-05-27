import random
import numpy as np
import torch

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Collate function for DataLoader (already inline in train.py, but here if needed)
def collate_fn(batch):
    return tuple(zip(*batch))

# Get device (cuda or cpu)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Draw boxes on image (optional visual helper, not required)
def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    import cv2
    img = img.permute(1, 2, 0).cpu().numpy().copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img
