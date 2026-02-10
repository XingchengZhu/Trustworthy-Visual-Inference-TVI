
import os
import glob

VAL_DIR = 'data/tiny-imagenet-200/val'
ANNOTATION_FILE = os.path.join(VAL_DIR, 'val_annotations.txt')
IMAGES_DIR = os.path.join(VAL_DIR, 'images')

def reorganize_val_images():
    # 1. Read annotations
    val_img_dict = {}
    with open(ANNOTATION_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_label = parts[1]
            val_img_dict[img_name] = class_label

    # 2. Organize
    if not os.path.exists(IMAGES_DIR):
        print(f"Images dir {IMAGES_DIR} not found. Trying flat structure in {VAL_DIR}...")
        # Check if images are in VAL_DIR directly or in VAL_DIR/images
        # Based on unzip, they are in tiny-imagenet-200/val/images/
        work_dir = os.path.join(VAL_DIR, 'images')
        if not os.path.exists(work_dir):
             print("Cannot find images. Exiting.")
             return
    else:
        work_dir = IMAGES_DIR

    print(f"Processing images in {work_dir}...")
    
    # Create class folders
    paths = glob.glob(os.path.join(work_dir, '*.JPEG'))
    print(f"Found {len(paths)} images.")
    
    for path in paths:
        filename = os.path.basename(path)
        if filename in val_img_dict:
            wnid = val_img_dict[filename]
            target_dir = os.path.join(work_dir, wnid)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, filename)
            os.rename(path, new_path)
            
    # Move class folders UP to val/ so ImageFolder can see val/class/image
    # Currently we have val/images/class/image
    # We want val/class/image
    
    # Actually, standard ImageFolder on 'val' expects 'val/class/image'.
    # So we should move headers from 'val/images/class' to 'val/class'.
    
    print("Moving class folders to root val directory...")
    class_dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    
    for cls in class_dirs:
        src = os.path.join(work_dir, cls)
        dst = os.path.join(VAL_DIR, cls)
        os.rename(src, dst)
        
    print("Done. Cleaning up empty images folder...")
    if not os.listdir(work_dir):
        os.rmdir(work_dir)
    print("TinyImageNet Validation Data Reorganized!")

if __name__ == '__main__':
    reorganize_val_images()
