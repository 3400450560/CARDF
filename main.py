import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 训练  
    model = YOLO(r"----------------")
    model.train(data=r"--------",batch=8,epochs=400,workers=8,close_mosaic=0,amp=False,
                optimizer='SGD',)

