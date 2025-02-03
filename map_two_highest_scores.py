import torch
from pycocotools.coco import COCO
import cv2
import numpy as np
from collections import defaultdict
import time
import torchvision.ops as ops

# Инициализация
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to(device)
model.eval()

# Загрузка COCO
annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)
imgIds = coco.getImgIds()  # Для обработки всего набора данных

# Сопоставление классов YOLO -> COCO
yolo_to_coco = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}


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


def process_image(img_id):
    img = coco.loadImgs(img_id)[0]
    img_path = f'val2017/{img["file_name"]}'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Предсказание
    results = model(img)
    preds = results.pred[0]
    if preds is None or len(preds) == 0:
        return defaultdict(list)

    # Обработка предсказаний
    boxes = preds[:, :4]
    scores = preds[:, 4]
    labels = preds[:, 5].long()

    # Выбираем два наиболее вероятных класса для каждого предсказания
    num_preds = len(scores)  # Количество всех предсказанных объектов
    top_scores, top_indices = torch.topk(scores, num_preds, largest=True)

    top_boxes = boxes[top_indices]
    top_scores = top_scores
    top_labels = labels[top_indices]

    # Ground Truth
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Если нет аннотаций для этого изображения, пропускаем его
    if len(anns) == 0:
        return defaultdict(list)

    gt_boxes = [np.array(ann['bbox']).reshape(1, 4) for ann in anns]
    gt_boxes = np.concatenate(gt_boxes)
    gt_boxes[:, 2:] += gt_boxes[:, :2]  # xywh -> xyxy
    gt_classes = [ann['category_id'] for ann in anns]

    # Перемещение всех тензоров на одно устройство (GPU или CPU)
    top_boxes = top_boxes.to(device)
    gt_boxes = torch.tensor(gt_boxes).to(device)

    # Сопоставление
    results = defaultdict(list)
    for box, label, score in zip(top_boxes, top_labels, top_scores):
        # Преобразование класса YOLO -> COCO
        coco_cls = yolo_to_coco.get(int(label), -1)
        if coco_cls == -1:
            continue

        # Вычисление вероятности для каждого класса
        class_scores = model(img).pred[0][:, 4:]  # Вероятности для всех классов

        # Сортируем вероятности по убыванию
        top_class_scores, top_class_indices = torch.topk(class_scores, 2, largest=True)

        for i in range(2):  # Проходим по двум наиболее вероятным классам
            class_idx = top_class_indices[0][i].item()  # Индекс наиболее вероятного класса
            class_score = top_class_scores[0][i].item()  # Вероятность этого класса

            # Вычисление IoU с Ground Truth
            ious = ops.box_iou(
                box.unsqueeze(0),  # Добавляем размерность батча
                gt_boxes
            ).cpu().numpy().flatten()

            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]
            gt_class = gt_classes[best_idx] if best_iou > 0.5 else -1

            # Запись TP/FP
            if gt_class == coco_cls:
                results[coco_cls].append(1)  # TP
            else:
                results[coco_cls].append(0)  # FP

    return results

# Основной цикл
all_results = defaultdict(list)
for i, img_id in enumerate(imgIds):
    res = process_image(img_id)
    for cls, vals in res.items():
        all_results[cls].extend(vals)


# Расчет mAP
def calculate_ap(recall, precision):
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        if np.any(mask):
            ap += np.max(precision[mask])
    return ap / 101

aps = []
print("\nClass-wise AP:")
for cls_id in coco.getCatIds():
    if cls_id not in all_results:
        continue

    tp = np.array(all_results[cls_id])
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    num_gt = len(coco.getAnnIds(catIds=cls_id))
    if num_gt == 0:
        continue

    recall = tp_cum / num_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)

    ap = calculate_ap(recall, precision)
    aps.append(ap)

    cls_name = coco.loadCats(cls_id)[0]['name']
    print(f"{cls_name:20} AP: {ap:.4f}")

mAP = np.mean(aps)
print(f"\nFinal mAP: {mAP:.4f}")
print(f"Time: {time.time() - start_time:.2f}s")