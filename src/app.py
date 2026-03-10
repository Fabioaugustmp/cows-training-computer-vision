from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import joblib
import pandas as pd
from ultralytics import YOLO
import io
import base64
import os
from PIL import Image

app = FastAPI(title="Cow ID Biometric System")

# Habilitar CORS para o frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuração de Modelos ---
# APONTANDO PARA O MODELO TREINADO (Substitua pelo caminho correto se necessário)
TRAINED_MODEL_PATH = '../runs/pose/runs/pose_kfold/cow_pose_kfold_1/weights/best.pt'

if not os.path.exists(TRAINED_MODEL_PATH):
    print(f"[AVISO] Modelo treinado não encontrado em {TRAINED_MODEL_PATH}")
    print("[AVISO] Tentando encontrar qualquer 'best.pt' na pasta runs...")
    # Busca automática caso o caminho acima mude
    import glob
    possible_models = glob.glob('../runs/**/best.pt', recursive=True)
    if possible_models:
        TRAINED_MODEL_PATH = possible_models[0]
        print(f"[OK] Modelo encontrado automaticamente: {TRAINED_MODEL_PATH}")

YOLO_MODEL = YOLO(TRAINED_MODEL_PATH) 
RF_ARTIFACTS = joblib.load('../models/cow_id_model.pkl')

CLF = RF_ARTIFACTS['model']
LE = RF_ARTIFACTS['label_encoder']

KPT = {
    'neck': 0, 'withers': 1, 'back': 2,
    'hook_l': 3, 'hook_r': 4, 'hip': 5,
    'tail': 6, 'pin_l': 7, 'pin_r': 8
}

def calculate_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 180.0
    arg = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.degrees(np.arccos(np.clip(arg, -1.0, 1.0))))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"\n{'='*60}")
    print(f" STEP 1: Image Acquisition & Decoding")
    print(f"{'='*60}")
    print(f"[INFO] Receiving file: {file.filename}")
    
    # 1. Ler imagem
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print("[ERROR] Failed to decode image. Ensure it is a valid JPG/PNG.")
        return {"error": "Falha ao decodificar imagem."}
    
    h, w, _ = img.shape
    print(f"[INFO] Image decoded successfully: {w}x{h} pixels.")

    print(f"\n{'='*60}")
    print(f" STEP 2: YOLO Pose Estimation (The 'AI Eye')")
    print(f"{'='*60}")
    print(f"[INFO] Running YOLOv8-Pose with confidence threshold 0.25...")
    # 2. YOLO Pose - Usando imgsz=640 para bater com o treino
    results = YOLO_MODEL.predict(img, conf=0.25, imgsz=640, verbose=False) 
    
    boxes = results[0].boxes
    print(f"[INFO] Found {len(boxes)} object(s) in the frame.")
    
    found_cow = False
    kpts = None
    
    # Percorre todas as detecções (boxes)
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"[DEBUG] Detection {i}: Class Index {cls}, Confidence: {conf:.2f}")
        
        # Verifica se essa detecção específica tem keypoints
        current_kpts = results[0].keypoints.data[i].cpu().numpy()
        
        # Se houver pontos e eles não forem todos zero
        if np.any(current_kpts[:, :2] > 0):
            print(f"[SUCCESS] Valid skeletal keypoints found for detection {i}!")
            kpts = current_kpts
            found_cow = True
            break

    if not found_cow:
        print("[ERROR] Object detected but keypoint extraction failed.")
        return {"error": "O modelo detectou o objeto, mas não conseguiu extrair o esqueleto (keypoints). Tente uma imagem mais clara."}

    print(f"\n{'='*60}")
    print(f" STEP 3: Anatomical Keypoint Mapping")
    print(f"{'='*60}")
    # 3. Mapeamento de pontos (x, y)
    points = {name: kpts[idx][:2] for name, idx in KPT.items()}
    print(f"[INFO] Extracted {len(points)} anatomical landmarks:")
    for name, pos in points.items():
        print(f"  - {name.ljust(10)}: x={int(pos[0])}, y={int(pos[1])}")

    print(f"\n{'='*60}")
    print(f" STEP 4: Geometric Biometrics (The 'Digital Tailor')")
    print(f"{'='*60}")
    # 4. Cálculo Biométrico
    try:
        body_len = calculate_dist(points['neck'], points['tail'])
        h_width = calculate_dist(points['hook_l'], points['hook_r'])
        p_width = calculate_dist(points['pin_l'], points['pin_r'])

        print(f"[INFO] Calculating distances for normalization (Reference Body Length: {body_len:.2f}px)")
        
        features = {
            'pelvic_ratio': float(h_width / p_width if p_width > 0 else 0),
            'body_aspect': float(body_len / h_width if h_width > 0 else 0),
            'spine_prop_1': float(calculate_dist(points['neck'], points['withers']) / calculate_dist(points['withers'], points['back'])),
            'spine_prop_2': float(calculate_dist(points['withers'], points['back']) / calculate_dist(points['back'], points['hip'])),
            'symmetry_idx': float(abs(calculate_dist(points['back'], points['hook_l']) - calculate_dist(points['back'], points['hook_r'])) / h_width),
            'angle_withers': calculate_angle(points['neck'], points['withers'], points['back']),
            'angle_back': calculate_angle(points['withers'], points['back'], points['hip']),
            'angle_hip': calculate_angle(points['back'], points['hip'], points['tail']),
            'norm_hook_width': float(h_width / body_len),
            'norm_pin_width': float(p_width / body_len),
            'avg_kpt_conf': float(np.mean(kpts[:, 2]))
        }
        
        print(f"[INFO] Biometric Signature generated (normalized ratios and angles):")
        print(f"  - Pelvic Ratio (Hook/Pin): {features['pelvic_ratio']:.3f}")
        print(f"  - Symmetry Index:          {features['symmetry_idx']:.3f}")
        print(f"  - Back Angle:             {features['angle_back']:.1f}°")

    except Exception as e:
        print(f"[ERROR] Biometric calculation failure: {str(e)}")
        return {"error": f"Erro no cálculo biométrico: {str(e)}"}

    print(f"\n{'='*60}")
    print(f" STEP 5: Machine Learning Classification")
    print(f"{'='*60}")
    # 5. Predição Random Forest - ORDEM DAS COLUNAS CRÍTICA
    feature_order = [
        'pelvic_ratio', 'body_aspect', 'spine_prop_1', 'spine_prop_2', 
        'symmetry_idx', 'angle_withers', 'angle_back', 'angle_hip', 
        'norm_hook_width', 'norm_pin_width', 'avg_kpt_conf'
    ]
    
    df_feat = pd.DataFrame([features])[feature_order]
    print(f"[INFO] Feeding feature vector into Random Forest model...")
    
    pred_idx = int(CLF.predict(df_feat)[0])
    cow_id = LE.inverse_transform([pred_idx])[0]
    
    if hasattr(cow_id, 'item'): cow_id = cow_id.item()
    if isinstance(cow_id, (np.integer, int)): cow_id = str(cow_id)

    probabilities = CLF.predict_proba(df_feat)[0]
    prob = float(probabilities[pred_idx])
    
    print(f"[SUCCESS] Classification Result: COW ID {cow_id} (Confidence: {prob:.2%})")

    print(f"\n{'='*60}")
    print(f" STEP 6: Visualization Synthesis")
    print(f"{'='*60}")
    print(f"[INFO] Rendering skeleton, labels and ID onto output image...")

    return {
        "cow_id": cow_id,
        "confidence": round(float(prob), 2),
        "biometrics": features,
        "image": img_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
