from torchvision.datasets import VOCDetection
from torchvision import transforms
import torch

class VOCDataset(VOCDetection):
    def __init__(self, root, year='2007', image_set='train', transform=None):
        super().__init__(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Convert to tensor
        if self.transform:
            img = self.transform(img)

        # Extract bounding boxes and labels
        boxes = []
        labels = []
        objs = target['annotation']['object']
        if not isinstance(objs, list):  # single object case
            objs = [objs]
        for obj in objs:
            bbox = obj['bndbox']
            box = [float(bbox['xmin']), float(bbox['ymin']),
                   float(bbox['xmax']), float(bbox['ymax'])]
            boxes.append(box)
            labels.append(1)  # single-class for demo

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return img, target
