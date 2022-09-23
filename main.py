from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import Response
from pathlib import Path
import torch
import io
import PIL.Image as Image

app = FastAPI()

label = []
name = []
save_dir = Path(r"D:\Work")
model_dir = Path(r'C:\Users\ADMIN\Downloads\Compressed\exp\weights\best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_dir)  # local model
img = []

@app.post("/files")
async def UploadImage(image: UploadFile = File(...)):
    name.append(image.filename)
    """with open(save_dir/name[-1],"wb") as img:
        shutil.copyfileobj(image.file,img)"""
    img.append(await image.read())
    return image.filename

@app.get("/files")
async def detect():
    image = Image.open(io.BytesIO(img[-1]))
    result = model(image)
    label.append(result)
    response = label[-1].pandas().xyxy[0].to_json(orient="records")
    return Response(response)
