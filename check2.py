import torch
from pycocotools.coco import COCO
import cv2
import os
import torchvision.ops as ops
import numpy as np
from collections import defaultdict

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)

# Список изображений
imgIds = coco.getImgIds()

# Функция для вычисления IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Оценка изображения с NMS
def evaluate_image_with_nms(img_id):
    img = coco.loadImgs(img_id)[0]
    img_path = f'val2017/{img["file_name"]}'
    img_cv = cv2.imread(img_path)

    # Из BGR в RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    model.eval()

    with torch.no_grad():
        # Предсказания YOLO
        results = model(img_rgb)

        predictions = results.pred[0]

        # Список классов для меток
        class_names = model.names

        # Реальные аннотации для этого изображения (ground truth)
        annIds = coco.getAnnIds(imgIds=img_id)  # Получаем id аннотаций для изображения
        anns = coco.loadAnns(annIds)  # Загружаем аннотации

        boxes = predictions[:, :4]  # Координаты bounding boxes [x1, y1, x2, y2]
        scores = predictions[:, 4]  # Уверенность предсказания
        labels = predictions[:, 5]  # Классы

        # NMS
        nms_indices = ops.nms(boxes, scores, 0.4)  # 0.4 - порог IoU для NMS
        nms_boxes = boxes[nms_indices]
        nms_scores = scores[nms_indices]
        nms_labels = labels[nms_indices]

        # Словарь для хранения IoU для каждого объекта
        iou_results = defaultdict(list)

        # Сохраняем предсказания и ground truth для дальнейшей оценки
        for ann in anns:
            gt_box = ann['bbox']  # Координаты [x, y, width, height]
            gt_x1, gt_y1, gt_width, gt_height = gt_box
            gt_x2 = gt_x1 + gt_width
            gt_y2 = gt_y1 + gt_height
            gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]  # Преобразуем в формат [x1, y1, x2, y2]
            iou_results['gt'].append((gt_box, ann['category_id']))

        for i, pred in enumerate(nms_boxes):
            x1, y1, x2, y2 = map(int, pred)
            confidence = nms_scores[i].item()
            label = int(nms_labels[i].item())
            pred_box = [x1, y1, x2, y2]

            max_iou = 0
            best_gt = None
            for gt_box, gt_category in iou_results['gt']:
                iou = compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt_category

            # Записываем результаты
            if max_iou >= 0.5:  # Точный порог IoU для определения правильного предсказания
                iou_results[label].append(1)  # True positive
            else:
                iou_results[label].append(0)  # False positive

    return iou_results

# Процесс по всем изображениям в папке val2017
all_results = defaultdict(list)

for img_id in imgIds:
    img_results = evaluate_image_with_nms(img_id)
    for label, results in img_results.items():
        if label != 'gt':  # Только классы, а не "gt"
            all_results[label].extend(results)

# Подсчет mAP
def compute_map(results):
    aps = {}
    for label, result in results.items():
        tp = np.array(result)
        fp = 1 - tp
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recall = cumsum_tp / np.sum(tp)  # Total number of ground truth objects
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)

        # Интерполяция для расчета AP
        recall_interpolated = np.linspace(0, 1, 101)
        precision_interpolated = np.interp(recall_interpolated, recall[::-1], precision[::-1])

        ap = np.mean(precision_interpolated)
        aps[label] = ap

    mAP = np.mean(list(aps.values()))  # Среднее значение по всем классам
    return aps, mAP


# Вычисление mAP для всех классов
aps, mAP = compute_map(all_results)
print(f"Mean Average Precision (mAP): {mAP:.4f}")
for label, ap in aps.items():
    print(f"AP for class {label}: {ap:.4f}")