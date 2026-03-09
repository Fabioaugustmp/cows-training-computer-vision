import os
import json

# Keypoint mapping
KPT_MAP = {
    'neck': 0,
    'withers': 1,
    'back': 2,
    'hook_left': 3,
    'hook_right': 4,
    'hip ridge': 5,
    'tail head': 6,
    'pin_left': 7,
    'pin_right': 8
}

NUM_KPT = 9

def convert_to_yolo():
    label_dir = 'labels/train'
    target_dir = 'labels/train' # We will save .txt here
    json_dir = 'labels/json'
    
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
        
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt') or filename.endswith('.ipynb_checkpoints'):
            continue
        
        file_path = os.path.join(label_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            print(f"Skipping {filename} - not a valid JSON")
            continue
            
        # Get image name from task
        image_url = data['task']['data']['img']
        image_name = os.path.basename(image_url)
        # Remove UUID prefix from Label Studio (e.g. 9ce2b0c7-...)
        # Wait, the images in images/train/ ALREADY have this UUID in the filename.
        # Let's check a filename.
        
        results = data['result']
        if not results:
            continue
            
        img_w = results[0]['original_width']
        img_h = results[0]['original_height']
        
        kpts = [[0.0, 0.0, 0] for _ in range(NUM_KPT)]
        
        # Separate hooks and pins for sorting
        hooks = []
        pins = []
        others = []
        
        for res in results:
            label = res['value']['keypointlabels'][0]
            x = res['value']['x'] / 100.0
            y = res['value']['y'] / 100.0
            
            if label == 'hook':
                hooks.append({'x': x, 'y': y})
            elif label == 'pin':
                pins.append({'x': x, 'y': y})
            elif label in KPT_MAP:
                idx = KPT_MAP[label]
                kpts[idx] = [x, y, 2] # 2 = visible
            else:
                print(f"Unknown label {label} in {filename}")
                
        # Handle hooks (L/R)
        hooks.sort(key=lambda h: h['y'])
        if len(hooks) >= 1:
            kpts[KPT_MAP['hook_left']] = [hooks[0]['x'], hooks[0]['y'], 2]
        if len(hooks) >= 2:
            kpts[KPT_MAP['hook_right']] = [hooks[1]['x'], hooks[1]['y'], 2]
            
        # Handle pins (L/R)
        pins.sort(key=lambda p: p['y'])
        if len(pins) >= 1:
            kpts[KPT_MAP['pin_left']] = [pins[0]['x'], pins[0]['y'], 2]
        if len(pins) >= 2:
            kpts[KPT_MAP['pin_right']] = [pins[1]['x'], pins[1]['y'], 2]
            
        # Calculate BBox from keypoints
        labeled_x = [k[0] for k in kpts if k[2] > 0]
        labeled_y = [k[1] for k in kpts if k[2] > 0]
        
        if not labeled_x:
            continue
            
        x_min, x_max = min(labeled_x), max(labeled_x)
        y_min, y_max = min(labeled_y), max(labeled_y)
        
        # Add margin (e.g. 10%)
        w = x_max - x_min
        h = y_max - y_min
        x_min = max(0, x_min - 0.1 * w)
        x_max = min(1, x_max + 0.1 * w)
        y_min = max(0, y_min - 0.1 * h)
        y_max = min(1, y_max + 0.1 * h)
        
        # YOLO format: <class> <x_center> <y_center> <width> <height> <kpts...>
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # Map back to 0-1 (already are 0-1)
        
        line = f"0 {x_center} {y_center} {width} {height}"
        for k in kpts:
            line += f" {k[0]} {k[1]} {k[2]}"
            
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        with open(os.path.join(target_dir, txt_name), 'w') as f:
            f.write(line + '\n')
            
        # Move processed JSON to labels/json
        os.rename(file_path, os.path.join(json_dir, filename))
        
    print("Done!")

if __name__ == '__main__':
    convert_to_yolo()
