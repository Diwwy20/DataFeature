from fastapi import FastAPI, Request
import numpy as np
import cv2
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root(): 
    return {"message": "This is my api"}

@app.get("/api/image/preprocessing")
async def Preprocessing(data: Request):
    try:
        json_data = await data.json()
        img_data = json_data["img_base64"]
        data_split = img_data.split(',', 1)
        img_str = data_split[1]

        decode_image_data = base64.b64decode(img_str)

        decode_img = cv2.imdecode(np.frombuffer(decode_image_data, np.uint8), cv2.IMREAD_COLOR)

        WIDTH = 55
        HEIGHT = 32
        resized_img = cv2.resize(decode_img, (WIDTH, HEIGHT))

        image = np.expand_dims(resized_img, axis=0)
        normalized_image = image / 255.0
        
        data = np.array(normalized_image, dtype="float")

        return {"data": data.tolist()}
    
    except Exception as ex:
         return {"error": f"เกิดข้อผิดพลาด: {str(ex)}"}