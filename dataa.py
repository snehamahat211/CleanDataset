import os
import shutil
import random
from turtledemo.nim import COLOR
import cv2

SOURCE_DIRS=['Bike_Car Data','Bus Data']
FINAL_DIR='FinalDatasets'
CLASSES_TO_KEEP=[2]
MAX_TOTAL_IMAGES=6000
SPLIT_RATIOS={'train':0.7,'val':0.2,'test':0.1}
random.seed(42)

def is_blurry(image_path,threshold=100.0):
    image=cv2.imread(image_path)
    if image is None:
        return True
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lap_var=cv2.Laplacian(gray,cv2.CV_64F).var()
    return lap_var<threshold

def has_license_plate(label_path):
    if not os.path.exists(label_path):
        return False
    with open(label_path,'r') as f:
        lines=f.readlines()
        return any(int(line.strip().split()[0]) in CLASSES_TO_KEEP for line in lines)
def filter_labels_to_license_plate(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        parts = line.strip().split()
        if int(parts[0]) in CLASSES_TO_KEEP:
            parts[0] = '0'
            filtered.append(' '.join(parts) + '\n')
    return filtered

def collect_data():
    data = []

    for source_dir in SOURCE_DIRS:
        for split in ['train', 'val']:
            img_dir = os.path.join(source_dir, 'images', split)
            lbl_dir = os.path.join(source_dir, 'labels', split)
            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                continue

            for img_file in os.listdir(img_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + '.txt')

                if not is_blurry(img_path) and has_license_plate(lbl_path):
                    data.append((img_path, lbl_path))

    random.shuffle(data)
    return data[:MAX_TOTAL_IMAGES]
def split_data(data):
    train_size=int(len(data)*SPLIT_RATIOS['train'])
    val_size=int(len(data)*SPLIT_RATIOS['val'])

    return{
        'train':data[:train_size],
        'val':data[train_size:train_size+val_size],
        'test':data[train_size+ val_size:]
    }
def prepare_final_folders():
    for subset in ['train','val','test']:
        os.makedirs(os.path.join(FINAL_DIR, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DIR, 'labels', subset), exist_ok=True)
data = collect_data()
data_split = split_data(data)

for split in data_split:
    print(f"{split.upper()}: {len(data_split[split])} images")


def copy_to_final(data_split):
    for subset,items in data_split.items():
        for img_path,lbl_path in items:
            img_dest=os.path.join(FINAL_DIR,'images',subset, os.path.basename(img_path))
            lbl_dest = os.path.join(FINAL_DIR, 'labels', subset, os.path.basename(lbl_path))

            filtered_labels = filter_labels_to_license_plate(lbl_path)

            if filtered_labels:
                with open(lbl_dest, 'w') as out_f:
                    out_f.writelines(filtered_labels)
                shutil.copy2(img_path, img_dest)

if __name__ == "__main__":
    if os.path.exists(FINAL_DIR):
        print(f" Removing old dataset at '{FINAL_DIR}'...")
        shutil.rmtree(FINAL_DIR)

    print("Collecting and filtering data:")
    data = collect_data()

    print(f"Total usable images with license plates: {len(data)}")
    data_split = split_data(data)

    print(f"preparing dataset folders")
    prepare_final_folders()

    print(f"copying filtered images and labels")
    copy_to_final(data_split)

    print("Done! Your YOLOv8-ready dataset is in 'FinalDatasets/'")























