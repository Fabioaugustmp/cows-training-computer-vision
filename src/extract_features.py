import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import math

# --- Configuração ---
MODEL_PATH = '../runs/pose/runs/pose_kfold/cow_pose_kfold_1/weights/best.pt'
DATASET_PATH = Path('../dataset_classificação')
OUTPUT_CSV = '../results/cow_features.csv'
CONF_THRESHOLD = 0.6  # Confiança mínima para detecção da vaca
KPT_CONF_THRESHOLD = 0.5  # Confiança média mínima para os pontos (keypoints)

# Mapeamento de Keypoints (conforme seu modelo original)
KPT = {
    'neck': 0, 'withers': 1, 'back': 2,
    'hook_l': 3, 'hook_r': 4, 'hip': 5,
    'tail': 6, 'pin_l': 7, 'pin_r': 8
}

def calculate_dist(p1, p2):
    """Calcula a distância euclidiana entre dois pontos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(p1, p2, p3):
    """Calcula o ângulo no ponto p2 dados os pontos p1, p2 e p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 180.0
    arg = np.dot(v1, v2) / (norm1 * norm2)
    return np.degrees(np.arccos(np.clip(arg, -1.0, 1.0)))

def get_cow_id(img_path):
    """Extrai o ID da baia/vaca a partir do nome do arquivo ou pasta pai."""
    # Primeiro tenta pegar o nome da pasta pai (comum em datasets organizados por ID)
    parent_name = img_path.parent.name
    if parent_name.isdigit():
        return parent_name

    # Fallback para o nome do arquivo (legado)
    filename = img_path.name
    parts = filename.split('_')
    for part in parts:
        if 'baia' in part.lower():
            return part.split('.')[0]
    return "unknown"

def extract_features():
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return

    print(f"Carregando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    data = []
    image_files = list(DATASET_PATH.rglob('*.jpg')) + list(DATASET_PATH.rglob('*.png'))

    if not image_files:
        print(f"Nenhuma imagem encontrada em {DATASET_PATH}")
        return

    print(f"Processando {len(image_files)} imagens com Biometria Avançada...")

    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Progresso: {i}/{len(image_files)}")

        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            if len(r) == 0 or r.keypoints is None or r.keypoints.data.shape[0] == 0:
                continue

            kpts_data = r.keypoints.data[0].cpu().numpy()

            avg_kpt_conf = np.mean(kpts_data[:, 2])
            if avg_kpt_conf < KPT_CONF_THRESHOLD:
                continue

            # Mapeamento de pontos (x, y)
            points = {name: kpts_data[idx][:2] for name, idx in KPT.items()}

            # Comprimento base para normalização (Pescoço até Cauda)
            body_len = calculate_dist(points['neck'], points['tail'])
            if body_len < 10: continue

            # Larguras pélvicas
            h_width = calculate_dist(points['hook_l'], points['hook_r'])
            p_width = calculate_dist(points['pin_l'], points['pin_r'])

            # --- RATIOS BIOMÉTRICOS ---
            pelvic_ratio = h_width / p_width if p_width > 0 else 0
            body_aspect_ratio = body_len / h_width if h_width > 0 else 0

            # Proporções da Coluna
            n_to_w = calculate_dist(points['neck'], points['withers'])
            w_to_b = calculate_dist(points['withers'], points['back'])
            b_to_h = calculate_dist(points['back'], points['hip'])

            spine_ratio_1 = n_to_w / w_to_b if w_to_b > 0 else 0
            spine_ratio_2 = w_to_b / b_to_h if b_to_h > 0 else 0

            # Índice de Simetria
            dist_back_hl = calculate_dist(points['back'], points['hook_l'])
            dist_back_hr = calculate_dist(points['back'], points['hook_r'])
            symmetry_idx = abs(dist_back_hl - dist_back_hr) / h_width if h_width > 0 else 0

            # Ângulos de Curvatura (Saúde e Bem-estar)
            angle_w = calculate_angle(points['neck'], points['withers'], points['back'])
            angle_b = calculate_angle(points['withers'], points['back'], points['hip'])
            angle_h = calculate_angle(points['back'], points['hip'], points['tail'])

            data.append({
                'filename': img_path.name,
                'cow_id': get_cow_id(img_path),
                'pelvic_ratio': pelvic_ratio,
                'body_aspect': body_aspect_ratio,
                'spine_prop_1': spine_ratio_1,
                'spine_prop_2': spine_ratio_2,
                'symmetry_idx': symmetry_idx,
                'angle_withers': angle_w,
                'angle_back': angle_b,
                'angle_hip': angle_h,
                'norm_hook_width': h_width / body_len,
                'norm_pin_width': p_width / body_len,
                'avg_kpt_conf': avg_kpt_conf
            })

    # Salvamento dos dados
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSucesso: {len(df)} perfis bovinos extraídos.")
        print(f"Arquivo gerado: {OUTPUT_CSV}")
    else:
        print("\nNenhum dado de alta qualidade foi encontrado para salvar.")


if __name__ == "__main__":
    extract_features()