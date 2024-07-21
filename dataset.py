from torchvision.datasets import OxfordIIITPet
from PIL import Image

class PetDataset(OxfordIIITPet):
    def __init__(self, root="./data", split='trainval', image_transform=None, mask_transform=None, download=True):
        super().__init__(root=root, split=split, target_types='segmentation', download=download)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, idx):
        image_path, label_path = self._images[idx], self._segs[idx]
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path)

        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
        