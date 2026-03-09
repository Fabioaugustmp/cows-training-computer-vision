import os
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kfold_training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_dataset(images_train, images_val, labels_train, labels_val):
    """Validates that all images have corresponding labels."""
    logger.info("Validating dataset integrity...")
    
    img_dirs = [images_train, images_val]
    lbl_dirs = [labels_train, labels_val]
    
    total_images = 0
    total_labels = 0
    
    for img_dir, lbl_dir in zip(img_dirs, lbl_dirs):
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")
            
        images = list(img_dir.glob('*.jpg'))
        for img in images:
            label = lbl_dir / (img.stem + '.txt')
            if not label.exists():
                logger.warning(f"Missing label for image: {img.name}")
            else:
                total_labels += 1
        total_images += len(images)
    
    logger.info(f"Validation complete: Found {total_images} images and {total_labels} valid labels.")
    if total_images != total_labels:
        logger.error(f"Mismatch detected! {total_images - total_labels} images are missing labels.")
    
    return total_images

def setup_kfold_training():
    # 1. Configuration & Paths
    dataset_root = Path.cwd() / 'dataset'
    images_train = dataset_root / 'images' / 'train'
    images_val = dataset_root / 'images' / 'val'
    labels_train = dataset_root / 'labels' / 'train'
    labels_val = dataset_root / 'labels' / 'val'
    
    try:
        num_samples = validate_dataset(images_train, images_val, labels_train, labels_val)
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return

    # Collect all images
    all_images = sorted(list(images_train.glob('*.jpg')) + list(images_val.glob('*.jpg')))
    
    # 2. K-Fold Parameters
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Load original yaml template
    yaml_template = 'coco8-pose.yaml'
    if not os.path.exists(yaml_template):
        logger.error(f"Template {yaml_template} not found in root directory.")
        return
        
    with open(yaml_template, 'r') as f:
        orig_config = yaml.safe_load(f)
        
    # Create a temp directory for split files
    split_dir = Path.cwd() / 'kfold_splits_firt_gen'
    split_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting {k}-Fold Cross-Validation on {num_samples} samples.")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        logger.info(f"Processing Fold {fold + 1}/{k}")
        
        # Define fold-specific image lists
        fold_train_files = [all_images[i] for i in train_idx]
        fold_val_files = [all_images[i] for i in val_idx]
        
        train_list_path = split_dir / f'fold_{fold+1}_train.txt'
        val_list_path = split_dir / f'fold_{fold+1}_val.txt'
        
        with open(train_list_path, 'w') as f:
            for img_path in fold_train_files:
                f.write(f"{img_path.absolute()}\n")
                
        with open(val_list_path, 'w') as f:
            for img_path in fold_val_files:
                f.write(f"{img_path.absolute()}\n")
        
        # Create temporary YAML for this fold
        fold_yaml_path = split_dir / f'fold_{fold+1}.yaml'
        fold_config = {
            'path': str(Path.cwd()), # Root path for YOLO
            'train': str(train_list_path),
            'val': str(val_list_path),
            'kpt_shape': orig_config['kpt_shape'],
            'names': orig_config['names']
        }
        
        with open(fold_yaml_path, 'w') as f:
            yaml.dump(fold_config, f)
            
        # 3. Train Model
        logger.info(f"Training Fold {fold + 1}...")
        model = YOLO('yolo26m-pose.pt')
        
        model.train(
            data=str(fold_yaml_path),
            epochs=100, # Balanced epochs for k-fold
            imgsz=640,
            device='0',
            name=f'cow_pose_kfold_{fold+1}',
            project='runs/pose_kfold',
            workers=8,
            batch=6,
            exist_ok=True,
            pretrained=True,
            cache=False
        )
        logger.info(f"Fold {fold + 1} completed.")
        
    logger.info("K-Fold Cross-Validation Complete!")
    logger.info("Check 'runs/pose_kfold' for metrics and weights.")

if __name__ == "__main__":
    setup_kfold_training()
