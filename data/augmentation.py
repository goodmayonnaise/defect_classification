'''
data/
 ├─ train/
 │   ├─ defect/
 │   │   ├─ img001.png
 │   │   ├─ img001_aug1.png
 │   │   ├─ img001_aug2.png
 │   └─ false/

'''


import os
from PIL import Image
from torchvision import transforms

def save_augmented_images(src, dst, transform, num_aug=5):
    os.makedirs(dst, exist_ok=True)
    
    for fname in os.listdir(src):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        img_path = os.path.join(src, fname)
        img = Image.open(img_path).convert("RGB")

        name, ext = os.path.splitext(fname)
        
        for i in range(num_aug):
            aug_img = transform(img)
            save_name = f"{name}_aug{i}{ext}"
            aug_img.save(os.path.join(dst, save_name))

        img.save(os.path.join(dst, fname))

    

if __name__ == "__main__":
    img_size = 300
    
    # save_transform = transforms.Compose([
    #     transforms.CenterCrop(img_size),
    #     transforms.RandomApply([
    #         transforms.RandomAffine(degrees=0, translate=(0.02,0.02), scale=(0.95,1.05))
    #     ], p=0.7),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(brightness=0.15, contrast=0.15)
    #     ], p=0.5),
    #     transforms.RandomApply([
    #         transforms.GaussianBlur(kernel_size=3)
    #     ], p=0.3)

    # ])

    # save_augmented_images(src=r"./data/split2/train/defect", 
    #                       dst=r"./data/aug2/train/defect",
    #                       transform=save_transform,
    #                       num_aug=5)
    

    # low mag defect 
    low_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.2)
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05))
        ], p=0.5)
    ])

    save_augmented_images(
        src=r"./data/low1/train/defect",
        dst=r"./data/low1_aug/train/defect",
        transform=low_transforms, 
        num_aug=5
    )    
    
    # high mag defect 
    high_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        ], p=0.7),

        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=1)
        ], p=0.3)
    ])
    
    save_augmented_images(
        src=r"./data/high1/train/defect",
        dst=r"./data/high1_aug1/train/defect",
        transform=high_transforms,
        num_aug=5
    )
