import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import math

# --- Configuração ---
# Utilizando os pesos do Fold 1 como modelo representante
MODEL_PATH = 'runs/pose/runs/pose_kfold/cow_pose_kfold_1/weights/best.pt'
DATASET_PATH = Path('dataset/images')
OUTPUT_CSV = 'cow_features.csv'

# Mapeamento de Keypoints conforme o seu modelo
KPT = {
    'neck': 0, 'withers': 1, 'back': 2,
    'hook_l': 3, 'hook_r': 4, 'hip': 5,
    'tail': 6, 'pin_l': 7, 'pin_r': 8
}


def calculate_angle(p1, p2, p3):
    """Calcula o ângulo no ponto p2 dados p1, p2 e p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    arg = np.dot(v1, v2) / (norm1 * norm2)
    arg = np.clip(arg, -1.0, 1.0)
    return np.degrees(np.arccos(arg))


def get_cow_id(filename):
    """Extrai o cow_id do nome do arquivo (Ex: baia16)."""
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
    image_files = list(DATASET_PATH.rglob('*.jpg'))

    if not image_files:
        print(f"Nenhuma imagem encontrada em {DATASET_PATH}")
        return

    print(f"Processando {len(image_files)} imagens...")

    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Progresso: {i}/{len(image_files)}")

        # Executa a predição
        results = model.predict(img_path, conf=0.6, verbose=False)

        for r in results:
            # CORREÇÃO DO INDEXERROR: Verifica se há detecções e se keypoints existem
            if len(r) == 0 or r.keypoints is None or r.keypoints.data.shape[0] == 0:
                continue

            # Extrai os keypoints da primeira vaca detectada com segurança
            # Usamos .cpu() antes do .numpy() para garantir compatibilidade
            kpts = r.keypoints.data[0].cpu().numpy()

            # Criamos um dicionário de pontos (x, y)
            points = {name: kpts[idx][:2] for name, idx in KPT.items()}

            # --- FEATURE 1: Ângulos (Curvatura) ---
            withers_angle = calculate_angle(points['neck'], points['withers'], points['back'])
            back_angle = calculate_angle(points['withers'], points['back'], points['hip'])
            hip_angle = calculate_angle(points['back'], points['hip'], points['tail'])

            # --- FEATURE 2: Distâncias Normalizadas (Proporções) ---
            # Fator de normalização: Comprimento do corpo (Pescoço até a Cauda)
            body_length = np.linalg.norm(points['neck'] - points['tail'])
            if body_length == 0: body_length = 1.0

            # Larguras Relativas
            hook_width = np.linalg.norm(points['hook_l'] - points['hook_r']) / body_length
            pin_width = np.linalg.norm(points['pin_l'] - points['pin_r']) / body_length

            # Comprimentos de Segmentos Relativos
            neck_to_withers = np.linalg.norm(points['neck'] - points['withers']) / body_length
            withers_to_back = np.linalg.norm(points['withers'] - points['back']) / body_length
            back_to_hip = np.linalg.norm(points['back'] - points['hip']) / body_length
            hip_to_tail = np.linalg.norm(points['hip'] - points['tail']) / body_length

            data.append({
                'filename': img_path.name,
                'cow_id': get_cow_id(img_path.name),
                'angle_withers': withers_angle,
                'angle_back': back_angle,
                'angle_hip': hip_angle,
                'dist_hook_width': hook_width,
                'dist_pin_width': pin_width,
                'seg_neck_withers': neck_to_withers,
                'seg_withers_back': withers_to_back,
                'seg_back_hip': back_to_hip,
                'seg_hip_tail': hip_to_tail
            })

    # Geração do arquivo final
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nExtração concluída. Total de registros: {len(df)}")
        print(f"Arquivo salvo: {OUTPUT_CSV}")
    else:
        print("\nNenhuma detecção válida foi realizada para gerar o CSV.")


if __name__ == "__main__":
    extract_features()