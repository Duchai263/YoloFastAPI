from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import Response
from pathlib import Path
import torch
import io
import PIL.Image as Image

app = FastAPI()


save_dir = Path(r"D:\Work")
model_dir = Path(r'C:\Users\ADMIN\Downloads\Compressed\exp\weights\best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_dir)  # local model


@app.post("/files")
async def UploadImage(image: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await image.read()))
    result = model(image)
    response = result.pandas().xyxy[0].to_json(orient="records")
    return Response(response)

@app.post("/MultipleFiles", response_class=PlainTextResponse)
async def UploadMultipleImages(images: list[UploadFile]):
    result = []
    response = []
    for each in images:
        image =  Image.open(io.BytesIO(await each.read()))
        result.append(model(image))
        response.append(result[-1].pandas().xyxy[0].to_json(orient="records"))
    
    result_text =''
    for each in response:
        result_text = result_text + str(each) + '\n'

    return result_text

#convert to 4 points
def toxy(result):
    pt1 = [result[0],result[1]]
    pt2 = [result[0],result[3]]
    pt3 = [result[2],result[1]]
    pt4 = [result[2],result[3]]
    convert = [ pt1, pt2, pt3, pt4, result[4], result[5], result[6] ]
    return convert

def ConvertTo4Pts(results):
    a = (results.pandas().xyxy[0]).to_numpy()
    b = []
    
    for each in a:
        b.append(toxy(each))
    
    df = pd.DataFrame(b)
    head = ['pt1','pt2','pt3','pt4','confidence','class','name']
    df = df.set_axis(head, axis= 'columns')
    return df

@app.post("/files/to4pts")
async def UploadImage(image: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await image.read()))
    result = model(image)
    response = ConvertTo4Pts(result).to_json(orient="records")
    return Response(response)

@app.post("/MultipleFiles/to4pts", response_class=PlainTextResponse)
async def UploadMultipleImages(images: list[UploadFile]):
    result = []
    response = []
    for each in images:
        image =  Image.open(io.BytesIO(await each.read()))
        result.append(model(image))
        response.append(ConvertTo4Pts(result[-1]).to_json(orient="records"))
    
    result_text =''
    for each in response:
        result_text = result_text + str(each) + '\n'

    return result_text
