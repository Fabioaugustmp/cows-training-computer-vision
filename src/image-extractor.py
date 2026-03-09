import os
import json
import random
import shutil
import logging

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("dataset_processing.log"), logging.StreamHandler()]
)

# --- CONFIGURATION ---
BASE_DATA_DIR = r'G:\.shortcut-targets-by-id\1xfU7Yl_DH9hYd36IT5RfJ1Quhm8ijZr8\Rotulação de Vacas'
DEST_ROOT = r'C:\Workspace\cows-train\dataset'
TRAIN_RATIO = 0.8

KPT_ORDER = [
    "neck", "withers", "back", "hook up", "hook down",
    "hip", "tail head", "pin up", "pin down"
]


def parse_label_studio_content(json_path):
    """Processes the internal JSON content into YOLO format."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        task = data if not isinstance(data, list) else data[0]
        results = task.get('result', [])
        if not results: return None

        # 1. BBox extraction
        bbox = next((item for item in results if item['type'] == 'rectanglelabels'), None)
        if not bbox: return None

        v = bbox['value']
        bx, by = (v['x'] + v['width'] / 2) / 100, (v['y'] + v['height'] / 2) / 100
        bw, bh = v['width'] / 100, v['height'] / 100

        # 2. Keypoints extraction
        kpts_dict = {}
        for item in results:
            if item['type'] == 'keypointlabels':
                name = item['value']['keypointlabels'][0]
                kx, ky = item['value']['x'] / 100, item['value']['y'] / 100

                # Visibility logic
                kpt_id = item['id']
                vis = 2
                vis_entry = next((i for i in results if i.get('id') == kpt_id and i['type'] == 'choices'), None)
                if vis_entry and "Oculto" in vis_entry['value']['choices']:
                    vis = 1
                kpts_dict[name] = f"{kx:.6f} {ky:.6f} {vis}"

        # 3. Final Assembly
        kpts_list = [kpts_dict.get(name, "0.000000 0.000000 0") for name in KPT_ORDER]
        return f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {' '.join(kpts_list)}\n"
    except Exception as e:
        logging.error(f"Failed to process content of {json_path}: {e}")
        return None


def main():
    # Clean Start
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)

    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(DEST_ROOT, sub), exist_ok=True)

    all_pairs = []
    logging.info(f"Searching for extensionless JSONs in: {BASE_DATA_DIR}")

    # Crawl folders (e.g., 01 - Thales, 02 - Fábio)
    for folder in os.listdir(BASE_DATA_DIR):
        folder_path = os.path.join(BASE_DATA_DIR, folder)
        if not os.path.isdir(folder_path): continue

        kp_dir = os.path.join(folder_path, "Key_points")
        if not os.path.exists(kp_dir): continue

        logging.info(f"Processing folder: {folder}")

        for filename in os.listdir(kp_dir):
            # Target files that are strictly numeric or have no extension
            if '.' in filename or filename.startswith('desktop'):
                continue

            label_path = os.path.join(kp_dir, filename)

            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    task = data if not isinstance(data, list) else data[0]
                    img_raw = task['task']['data']['img']

                    # Filename extraction (Handles %5C and UUIDs)
                    img_name = img_raw.replace('%5C', '/').split('/')[-1]
                    if '-' in img_name and len(img_name.split('-', 1)[0]) == 8:
                        img_name = img_name.split('-', 1)[-1]

                    full_img_path = os.path.join(folder_path, img_name)

                    if os.path.exists(full_img_path):
                        all_pairs.append((full_img_path, label_path, img_name))
                    else:
                        logging.warning(f"Match failed: {img_name} not found in {folder}")
            except Exception:
                continue

    if not all_pairs:
        logging.error("No pairs found. Verify that images are in the parent folder of 'Key_points'.")
        return

    # Shuffle and Split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * TRAIN_RATIO)

    datasets = {
        'train': all_pairs[:split_idx],
        'val': all_pairs[split_idx:]
    }

    for mode, pairs in datasets.items():
        count = 0
        for img_path, label_path, img_name in pairs:
            yolo_content = parse_label_studio_content(label_path)
            if yolo_content:
                # Copy Image
                shutil.copy2(img_path, os.path.join(DEST_ROOT, 'images', mode, img_name))
                # Write Label
                label_txt = os.path.splitext(img_name)[0] + ".txt"
                with open(os.path.join(DEST_ROOT, 'labels', mode, label_txt), 'w') as f:
                    f.write(yolo_content)
                count += 1
        logging.info(f"Saved {count} files to {mode} set.")


if __name__ == "__main__":
    main()