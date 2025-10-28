import os
import random
import shutil

# ----------------------------
# CONFIGURATION
# ----------------------------
train_folder = "../train/"   # folder that currently has ALL images
val_folder = "../val/"       # new/empty validation folder to create
val_ratio = 0.2              # 20% of each class goes to val

# for reproducibility: same images will always be chosen
random.seed(42)

# ----------------------------
# Create val folder if it doesn't exist
# ----------------------------
if not os.path.exists(val_folder):
    os.makedirs(val_folder)

# ----------------------------
# Split images class by class
# ----------------------------
for class_name in os.listdir(train_folder):
    class_train_path = os.path.join(train_folder, class_name)

    # skip anything that's not a directory
    if not os.path.isdir(class_train_path):
        continue

    class_val_path = os.path.join(val_folder, class_name)
    if not os.path.exists(class_val_path):
        os.makedirs(class_val_path)

    # list all files in this class
    images = [
        f for f in os.listdir(class_train_path)
        if os.path.isfile(os.path.join(class_train_path, f))
    ]

    # shuffle in-place
    random.shuffle(images)

    # how many go to val
    num_val = max(1, int(len(images) * val_ratio))
    val_images = images[:num_val]

    # move those images from train -> val
    for img in val_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)

    print(f"{class_name}: moved {num_val} to val, kept {len(images) - num_val} in train")

print("Dataset split complete!")