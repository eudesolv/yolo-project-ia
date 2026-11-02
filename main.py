"""
API REST para Treinamento, Comparação e Detecção de Pokémon com YOLOv8 e YOLOv11.
Autor: IA / Usuário
"""

import os
import shutil
import cv2
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from ultralytics import YOLO
from roboflow import Roboflow

app = FastAPI(
    title="Pokémon YOLOv8 vs YOLOv11 API",
    description="API REST para treinar, comparar e realizar inferências de detecção de objetos em Pokémon.",
    version="1.0.0"
)

# Caminhos globais para persistência de modelos e dados
DATASET_DIR = "./dataset"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Armazenamento em memória dos modelos carregados
loaded_models: Dict[str, YOLO] = {}


class DatasetRequest(BaseModel):
    roboflow_api_key: str
    workspace: str = "pokemon-detection"
    project: str = "pokemon-yolo"
    version: int = 1


class TrainRequest(BaseModel):
    roboflow_api_key: str
    epochs: int = 15
    imgsz: int = 640
    batch: int = 16


def download_dataset(api_key: str, workspace: str, project: str, version: int) -> str:
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=DATASET_DIR)
    return dataset.location


@app.get("/")
def home():
    return {
        "message": "API de Detecção de Pokémon rodando com sucesso!",
        "endpoints": {
            "POST /download-dataset": "Baixa o dataset via Roboflow",
            "POST /train-and-compare": "Treina e compara YOLOv8 e YOLOv11",
            "POST /predict/{model_version}": "Recebe uma imagem e retorna caixas delimitadoras"
        }
    }


@app.post("/download-dataset")
def api_download_dataset(req: DatasetRequest):
    try:
        path = download_dataset(req.roboflow_api_key, req.workspace, req.project, req.version)
        return {"status": "sucesso", "dataset_path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao baixar dataset: {str(e)}")


@app.post("/train-and-compare")
def train_and_compare(req: TrainRequest):
    """
    Baixa o dataset, treina YOLOv8n e YOLO11n, compara as métricas mAP@0.5 e carrega os modelos na memória.
    """
    try:
        dataset_path = download_dataset(req.roboflow_api_key, "pokemon-detection", "pokemon-yolo", 1)
        data_yaml = os.path.join(dataset_path, "data.yaml")

        if not os.path.exists(data_yaml):
            raise HTTPException(status_code=400, detail="Arquivo data.yaml não encontrado no dataset.")

        # 1. Treinamento YOLO
        print("[INFO] Treinando YOLOv8...")
        model_v8 = YOLO("yolov8n.pt")
        res_v8 = model_v8.train(data=data_yaml, epochs=req.epochs, imgsz=req.imgsz, batch=req.batch, name="yolov8_pokemon")
        v8_path = os.path.join(MODELS_DIR, "yolov8_best.pt")
        shutil.copy(model_v8.trainer.best, v8_path)
        loaded_models["yolov8"] = YOLO(v8_path)

        # 2. Treinamento YOLOv11
        print("[INFO] Treinando YOLOv11...")
        model_v11 = YOLO("yolo11n.pt")
        res_v11 = model_v11.train(data=data_yaml, epochs=req.epochs, imgsz=req.imgsz, batch=req.batch, name="yolo11_pokemon")
        v11_path = os.path.join(MODELS_DIR, "yolo11_best.pt")
        shutil.copy(model_v11.trainer.best, v11_path)
        loaded_models["yolo11"] = YOLO(v11_path)

        # 3. Validação e Comparação
        val_v8 = loaded_models["yolov8"].val(data=data_yaml)
        val_v11 = loaded_models["yolo11"].val(data=data_yaml)

        map50_v8 = float(val_v8.box.map50)
        map50_v11 = float(val_v11.box.map50)

        winner = "YOLOv11" if map50_v11 > map50_v8 else "YOLOv8"

        return {
            "status": "Treinamento e comparação concluídos",
            "metricas": {
                "YOLOv8_mAP50": round(map50_v8, 4),
                "YOLOv11_mAP50": round(map50_v11, 4),
                "modelo_vencedor": winner
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante treinamento/comparação: {str(e)}")


@app.post("/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):

    if model_version not in ["yolov8", "yolo11"]:
        raise HTTPException(status_code=400, detail="Versão de modelo inválida. Escolha 'yolov8' ou 'yolo11'.")

    if model_version not in loaded_models:
        model_file = os.path.join(MODELS_DIR, f"{model_version}_best.pt")
        if os.path.exists(model_file):
            loaded_models[model_version] = YOLO(model_file)
        else:            
            default_weights = "yolov8n.pt" if model_version == "yolov8" else "yolo11n.pt"
            loaded_models[model_version] = YOLO(default_weights)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = loaded_models[model_version]
        results = model.predict(source=img, save=False)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                detections.append({
                    "classe": class_name,
                    "confianca": round(confidence, 4),
                    "bounding_box": {
                        "x_min": round(x1, 2),
                        "y_min": round(y1, 2),
                        "x_max": round(x2, 2),
                        "y_max": round(y2, 2)
                    }
                })

        return {
            "modelo_utilizado": model_version,
            "total_deteccoes": len(detections),
            "deteccoes": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da imagem: {str(e)}")