import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np
import time

#модель YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

#датасет COCO
annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)

#список изображений для проверки
imgIds = coco.getImgIds()

#параметры для оценки
cocoGt = coco
cocoDt = []

start_time = time.time()
#функция для оценки на одном изображении
def evaluate_image(img_id):
    #путь к изображению
    img = coco.loadImgs(img_id)[0]
    img_path = f'val2017/{img["file_name"]}'

    #чтение изображения
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    #предсказания YOLO
    results = model(img_rgb)  #передаем изображение в модель
    predictions = results.pred[0]  #предсказания для первого изображения

    #преобразуем результаты в формат COCO
    cocoDt = []
    for pred in predictions:
        # (класс, уверенность, x1, y1, x2, y2)
        cocoDt.append({
            'image_id': img_id,
            'category_id': int(pred[5]),
            'bbox': [float(pred[0]), float(pred[1]), float(pred[2] - pred[0]), float(pred[3] - pred[1])],
            'score': float(pred[4])
        })
    return cocoDt

#сбор предсказаний для всех изображений
for img_id in imgIds:
    cocoDt.extend(evaluate_image(img_id))

#преобразуем предсказания в формат COCO
cocoDt = cocoGt.loadRes(cocoDt)

#оценка mAP
cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time) 
