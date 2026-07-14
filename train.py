import time
import warnings

# import os
# os.environ["WANDB_MODE"] = "offline"

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # # ins seg on coco
    model1 = YOLO('ultralytics/cfg/models/v8-seg-comp/yolov8s-seg-p6-EdgeUp.yaml').to(device=0)
    # model1 = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml').to(device=0)
    model1.train(data='ultralytics/cfg/datasets/coco.yaml',
                 imgsz=640,
                 epochs=300,
                 batch=16,  ##
                 task='segment',
                 lr0=0.01,  ##
                 patience=50,
                 optimizer='SGD',
                 workers=8,
                 pretrained=True,
                 )
    # detect on COCO
    # model1 = YOLO('ultralytics/cfg/models/V8-define/yolov8s-EdgeUp.yaml').to(device=0)
    # # model1 = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml').to(device=0)
    # model1.train(data='ultralytics/cfg/datasets/coco.yaml',
    #              imgsz=640,
    #              epochs=300,
    #              batch=16,  #
    #              lr0=0.01,  ##
    #              patience=50,
    #              optimizer='SGD',
    #              workers=8,
    #              pretrained=True,
    #              )



