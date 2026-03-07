import json
import os

# MUST match your YAML exactly
KPT_ORDER = [
    "neck",       # Ensure this is labeled exactly as 'neck' in LS
    "withers",
    "back",
    "hook up",    # Matches your JSON
    "hook down",  # Matches your JSON
    "hip",        # Matches your JSON
    "tail head",  # Matches your JSON (ensure it's not 'tailhead')
    "pin up",     # Matches your JSON
    "pin down"    # Matches your JSON
]


def convert_all_files(input_folder, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    processed_count = 0
    for filename in all_files:
        try:
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            tasks = data if isinstance(data, list) else [data]
            for task in tasks:
                results = task.get('result', [])
                if not results: continue

                # --- 1. Smart Filename extraction ---
                img_full_path = task['task']['data']['img']

                # Check if it's a local file path (%5C) or a standard web/upload path (/)
                if '%5C' in img_full_path:
                    img_name = img_full_path.split('%5C')[-1]
                else:
                    # Get the filename after the last slash
                    img_name = os.path.basename(img_full_path)

                    # If it has the Label Studio UUID prefix (e.g., d2766c60-filename.jpg)
                    # We look for the first '-' after the prefix and take the rest.
                    # This safely handles: d2766c60-20260101_205224_baia20_IPC2.jpg
                    if '-' in img_name and len(img_name.split('-', 1)[0]) == 8:
                        img_name = img_name.split('-', 1)[-1]

                label_filename = os.path.splitext(img_name)[0] + ".txt"

                # 2. Get BBox from Label Studio (rectanglelabels)
                bbox = next((item for item in results if item['type'] == 'rectanglelabels'), None)
                if not bbox: continue

                v = bbox['value']
                bx, by = (v['x'] + v['width'] / 2) / 100, (v['y'] + v['height'] / 2) / 100
                bw, bh = v['width'] / 100, v['height'] / 100

                # 3. Get Keypoints
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

                # 4. Final Assembly (Ensuring exactly 9 points)
                kpts_list = [kpts_dict.get(name, "0.000000 0.000000 0") for name in KPT_ORDER]
                yolo_line = f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {' '.join(kpts_list)}\n"

                with open(os.path.join(output_labels_dir, label_filename), 'w') as f:
                    f.write(yolo_line)
                processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Success! Created {processed_count} labels.")


JSON_FOLDER = r'C:\Users\fabio\Pictures\COWS-DATASET\05 - Camilla (pronto)-20260304T143124Z-3-001\05 - Camilla (pronto)\Key_points'
OUTPUT_FOLDER = r'C:\Workspace\cows-train\labels\train'

if __name__ == "__main__":
    convert_all_files(JSON_FOLDER, OUTPUT_FOLDER)