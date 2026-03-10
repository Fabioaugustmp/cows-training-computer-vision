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
    print(f"\n[DEBUG] Recebido arquivo: {file.filename}")
    
    # 1. Ler imagem
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Falha ao decodificar imagem."}

    # 2. YOLO Pose - Usando imgsz=640 para bater com o treino
    results = YOLO_MODEL.predict(img, conf=0.25, imgsz=640, verbose=True) 
    
    print(f"[DEBUG] Número de detecções no frame: {len(results[0].boxes)}")
    
    found_cow = False
    kpts = None
    
    # Percorre todas as detecções (boxes)
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"[DEBUG] Detecção {i}: Classe={cls}, Confiança={conf:.2f}")
        
        # Verifica se essa detecção específica tem keypoints
        current_kpts = results[0].keypoints.data[i].cpu().numpy()
        
        # Se houver pontos e eles não forem todos zero
        if np.any(current_kpts[:, :2] > 0):
            print(f"[DEBUG] -> Keypoints encontrados para a detecção {i}!")
            kpts = current_kpts
            found_cow = True
            break

    if not found_cow:
        print("[DEBUG] Nenhuma detecção com keypoints válidos foi encontrada.")
        return {"error": "O modelo detectou o objeto, mas não conseguiu extrair o esqueleto (keypoints). Tente uma imagem mais clara."}

    # 3. Mapeamento de pontos (x, y)
    points = {name: kpts[idx][:2] for name, idx in KPT.items()}
    print(f"[DEBUG] Keypoints mapeados: {list(points.keys())}")

    # 4. Cálculo Biométrico
    try:
        body_len = calculate_dist(points['neck'], points['tail'])
        h_width = calculate_dist(points['hook_l'], points['hook_r'])
        p_width = calculate_dist(points['pin_l'], points['pin_r'])

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
    except Exception as e:
        return {"error": f"Erro no cálculo biométrico: {str(e)}"}

    # 5. Predição Random Forest - ORDEM DAS COLUNAS CRÍTICA
    # Criamos o DataFrame garantindo que as colunas estejam EXATAMENTE na ordem do treino
    feature_order = [
        'pelvic_ratio', 'body_aspect', 'spine_prop_1', 'spine_prop_2', 
        'symmetry_idx', 'angle_withers', 'angle_back', 'angle_hip', 
        'norm_hook_width', 'norm_pin_width', 'avg_kpt_conf'
    ]
    
    # Criar DataFrame com ordem fixa
    df_feat = pd.DataFrame([features])[feature_order]
    
    print(f"[DEBUG] Vetor de Entrada: {df_feat.values}")
    
    pred_idx = int(CLF.predict(df_feat)[0])
    cow_id = LE.inverse_transform([pred_idx])[0]
    
    # Se cow_id for um tipo numpy (como np.int64), converta para Python nativo
    if hasattr(cow_id, 'item'):
        cow_id = cow_id.item()
    elif isinstance(cow_id, np.generic):
        cow_id = cow_id.tolist()
    # Caso seja string vinda de pasta (dataset_classificação), já está ok, 
    # mas garantimos convertendo para str se for o caso de IDs numéricos.
    if isinstance(cow_id, (np.integer, int)):
        cow_id = str(cow_id)

    probabilities = CLF.predict_proba(df_feat)[0]
    prob = float(probabilities[pred_idx])

    # 6. Desenhar Esqueleto (Visual aprimorado)
    # Aumentar espessura para screenshots de alta resolução
    thickness = max(2, int(img.shape[1] / 400)) 
    
    # Esqueleto seguindo o padrão da imagem do problema
    skeleton = [
        ('neck', 'withers'), ('withers', 'back'), ('back', 'hip'), ('hip', 'tail'),
        ('hook_l', 'pin_l'), ('hook_r', 'pin_r'),
        ('hook_l', 'hook_r'), ('pin_l', 'pin_r'),
        ('back', 'hook_l'), ('back', 'hook_r')
    ]
    
    # Cores para facilitar identificação
    colors = {
        'neck': (0, 0, 255),      # Vermelho
        'withers': (0, 165, 255), # Laranja
        'back': (0, 255, 255),    # Amarelo
        'hip': (0, 255, 0),       # Verde
        'tail': (255, 0, 0),      # Azul
        'hook_l': (255, 0, 255),  # Magenta
        'hook_r': (255, 0, 255),
        'pin_l': (255, 255, 0),   # Ciano
        'pin_r': (255, 255, 0)
    }

    # Desenhar linhas do esqueleto (somente se pontos forem válidos)
    for start_node, end_node in skeleton:
        if start_node in points and end_node in points:
            p1 = (int(points[start_node][0]), int(points[start_node][1]))
            p2 = (int(points[end_node][0]), int(points[end_node][1]))
            # Evitar desenhar linhas para a origem (0,0) que indicam pontos não detectados
            if p1[0] > 1 and p1[1] > 1 and p2[0] > 1 and p2[1] > 1:
                cv2.line(img, p1, p2, (200, 200, 200), thickness)

    # Desenhar pontos e nomes com diagnóstico
    for i, (name, pt) in enumerate(points.items()):
        x, y = int(pt[0]), int(pt[1])
        if x > 1 and y > 1:
            color = colors.get(name, (0, 255, 0))
            conf = kpts[i, 2] # Confiança do keypoint vinda do YOLO
            
            # Desenhar círculo do ponto
            cv2.circle(img, (x, y), thickness + 3, color, -1)
            
            # Rótulo diagnóstico: "0. neck (0.85)"
            diag_label = f"{i}. {name} ({conf:.2f})"
            
            # Fundo preto para o texto (melhor leitura)
            (w, h), _ = cv2.getTextSize(diag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, 1)
            cv2.rectangle(img, (x + 10, y - h - 5), (x + 10 + w, y + 5), (0,0,0), -1)
            
            cv2.putText(img, diag_label, (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, color, 1)
    
    # Adicionar texto do ID na imagem
    cv2.putText(img, f"ID: {cow_id} ({prob:.2f})", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, thickness, (0, 255, 0), thickness + 1)
    
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "cow_id": cow_id,
        "confidence": round(float(prob), 2),
        "biometrics": features,
        "image": img_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
