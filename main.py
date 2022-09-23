from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import Response
from pathlib import Path
import torch
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()
model_dir = Path(r'C:\Users\ADMIN\Downloads\Compressed\exp\weights\best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_dir)  # local model

label = []
name = []
save_dir = Path(r"D:\Work")

@app.post("/files")
async def UploadImage(file: UploadFile = File(...)):
    name.append(file.filename)
    with open(save_dir/name[-1],"wb") as img:
        shutil.copyfileobj(file.file,img)
    return file.filename

@app.get("/files")
async def results():
    img = save_dir/name[-1]
    label.append(model(img))
    json = label[-1].pandas().xyxy[0].to_json(orient="records")
    return json
