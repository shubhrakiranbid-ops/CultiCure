import os
import shutil
import random

# Parameters
source_dir = 'all_data'  # Folder where all your class folders are
target_dir = 'dataset'
train_ratio = 0.8  # 80% train, 20% val

# Create target directories
for split in ['train', 'val']:
    split_path = os.path.join(target_dir, split)
    os.makedirs(split_path, exist_ok=True)

# For each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create class folders in train/val
    for split, split_images in zip(['train', 'val'], [train_images, val_images]):
        split_class_dir = os.path.join(target_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy2(src, dst)

print("Dataset split complete!")