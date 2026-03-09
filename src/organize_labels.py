import os
import json

# YOLO Pose Mapping (based on coco8-pose.yaml)
# 0: neck
# 1: withers
# 2: back
# 3: hook_left
# 4: hook_right
# 5: hip_ridge
# 6: tail_head
# 7: pin_left
# 8: pin_right

KPT_MAP = {
    'withers': 1,
    'back': 2,
    'hook up': 3,
    'hook down': 4,
    'hip': 5,
    'tail head': 6,
    'pin up': 7,
    'pin down': 8
}

def get_image_map():
    image_map = {}
    for split in ['train', 'val']:
        img_dir = os.path.join('../data/dataset/images', split)
        if not os.path.exists(img_dir):
            continue
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Key is the suffix after the first dash
                # 9ce2b0c7-20260101_205224_baia20_IPC2.jpg -> 20260101_205224_baia20_IPC2.jpg
                parts = filename.split('-', 1)
                if len(parts) > 1:
                    suffix = parts[1]
                else:
                    suffix = filename
                image_map[suffix] = {
                    'full_name': filename,
                    'split': split
                }
    return image_map

def organize_labels():
    key_points_dir = '../data/Key_points'
    image_map = get_image_map()
    
    if not os.path.exists('../data/labels/train'):
        os.makedirs('../data/labels/train')
    if not os.path.exists('../data/labels/val'):
        os.makedirs('../data/labels/val')
        
    converted_count = 0
    missing_count = 0
    
    for filename in os.listdir(key_points_dir):
        # The filenames are task IDs like '310', no extension
        file_path = os.path.join(key_points_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                print(f"Skipping {filename} - invalid JSON")
                continue
        
        # Get image suffix from task data
        # "/data/upload/1/d2766c60-20260101_205224_baia20_IPC2.jpg"
        img_url = data['task']['data']['img']
        img_basename = os.path.basename(img_url)
        img_parts = img_basename.split('-', 1)
        img_suffix = img_parts[1] if len(img_parts) > 1 else img_basename
        
        if img_suffix not in image_map:
            print(f"Image suffix {img_suffix} not found in images/ directory (from task {filename})")
            missing_count += 1
            continue
            
        img_info = image_map[img_suffix]
        target_dir = os.path.join('labels', img_info['split'])
        label_filename = os.path.splitext(img_info['full_name'])[0] + '.txt'
        label_path = os.path.join(target_dir, label_filename)
        
        results = data['result']
        
        # Initialize keypoints (9 kpts, each x, y, visibility)
        # x, y are normalized 0-1, visibility is 0 (missing), 1 (labeled but hidden), 2 (visible)
        kpts = [[0.0, 0.0, 0] for _ in range(9)]
        bbox = None # [x_center, y_center, width, height]
        
        for item in results:
            if item['type'] == 'keypointlabels':
                label = item['value']['keypointlabels'][0]
                if label in KPT_MAP:
                    idx = KPT_MAP[label]
                    # Label Studio coordinates are in percentages 0-100
                    x = item['value']['x'] / 100.0
                    y = item['value']['y'] / 100.0
                    kpts[idx] = [x, y, 2]
            elif item['type'] == 'rectanglelabels':
                # Label Studio bbox: x, y are top-left in percentages
                # x, y, width, height are all in 0-100
                val = item['value']
                w = val['width'] / 100.0
                h = val['height'] / 100.0
                x_tl = val['x'] / 100.0
                y_tl = val['y'] / 100.0
                x_center = x_tl + w / 2.0
                y_center = y_tl + h / 2.0
                bbox = [x_center, y_center, w, h]
        
        if bbox is None:
            # Fallback: calculate bbox from keypoints if cow rectangle is missing
            xs = [k[0] for k in kpts if k[2] > 0]
            ys = [k[1] for k in kpts if k[2] > 0]
            if xs:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w = (x_max - x_min) * 1.1 # 10% margin
                h = (y_max - y_min) * 1.1
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                bbox = [x_center, y_center, w, h]
            else:
                print(f"Skipping {filename} - no cow bbox or keypoints found")
                continue
        
        # Write YOLO format line
        # <class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_v> ...
        yolo_line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        for kp in kpts:
            yolo_line += f" {kp[0]:.6f} {kp[1]:.6f} {kp[2]}"
            
        with open(label_path, 'w') as lf:
            lf.write(yolo_line + '\n')
        
        converted_count += 1
        
    print(f"Successfully converted {converted_count} labels.")
    print(f"Missed {missing_count} images.")

if __name__ == '__main__':
    organize_labels()
