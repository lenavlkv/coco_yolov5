from ultralytics import YOLO
import torch
from pycocotools.coco import COCO
import numpy as np
from collections import defaultdict


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    """
    print(f"box1: {box1}, box2: {box2}")

    # Проверяем, что оба бокса - это списки или кортежи с 4 элементами
    if not (isinstance(box1, (list, tuple)) and len(box1) == 4):
        raise ValueError(f"Неверный формат box1: {box1}")
    if not (isinstance(box2, (list, tuple)) and len(box2) == 4):
        raise ValueError(f"Неверный формат box2: {box2}")

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)

    intersect_area = max(0, intersect_x2 - intersect_x1 + 1) * max(0, intersect_y2 - intersect_y1 + 1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersect_area / float(box1_area + box2_area - intersect_area)
    return iou


def evaluate_mAP(predictions, coco, iou_threshold=0.5):
    """
    Evaluate mAP for the predictions against ground truth annotations.
    """
    all_true_boxes = defaultdict(list)
    all_pred_boxes = defaultdict(list)

    # Собираем ground truth боксы по категориям
    for img_id, gt_boxes in coco.imgToAnns.items():
        for ann in gt_boxes:
            category_id = ann['category_id']
            all_true_boxes[category_id].append(ann['bbox'])

    # Собираем предсказания по категориям
    for pred in predictions:
        img_id = pred['image_id']
        for box in pred['bbox']:
            category_id = pred['best_cls']
            all_pred_boxes[category_id].append({
                'bbox': box,
                'score': pred['best_conf'],
                'image_id': img_id
            })

    # Для каждой категории считаем AP
    ap_per_class = {}
    for category_id, true_boxes in all_true_boxes.items():
        if category_id not in all_pred_boxes:
            continue

        pred_boxes = all_pred_boxes[category_id]
        pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)

        tp, fp = [], []
        matched_gt = set()

        for pred_box in pred_boxes:
            iou_max = 0
            matched_gt_id = -1

            # Сравниваем с каждым ground truth боксом и находим максимальный IoU
            for i, gt_box in enumerate(true_boxes):
                iou = calculate_iou(pred_box['bbox'], gt_box)
                if iou > iou_max:
                    iou_max = iou
                    matched_gt_id = i

            if iou_max >= iou_threshold and matched_gt_id not in matched_gt:
                tp.append(1)
                fp.append(0)
                matched_gt.add(matched_gt_id)
            else:
                tp.append(0)
                fp.append(1)

        # Вычисление precision, recall, и AP
        tp = np.array(tp)
        fp = np.array(fp)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / (len(true_boxes) + 1e-6)

        ap = np.trapz(precision, recall)
        ap_per_class[category_id] = ap

    # Вычисление mAP (среднее значение AP для всех классов)
    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class


def results_predict(img_path, model, threshold=0.5, iou=0.7):
    """
    Run prediction with a YOLO model and apply Non-Maximum Suppression (NMS) to the results.
    """
    # Запуск инференса
    results = model(img_path)

    # Извлекаем боксы из результатов
    boxes = []
    for result in results:
        for box in result.boxes.data.tolist():  # Предсказания для изображения
            print(f"Extracted box: {box}")  # Отладочный вывод
            # box должен содержать [x0, y0, x1, y1, conf, cls]
            if len(box) >= 6:  # Убедимся, что в боксе есть хотя бы 6 элементов
                x0, y0, x1, y1, conf, cls = box
                # Преобразуем в [x, y, width, height]
                boxes.append({
                    'image_id': img_path,
                    'bbox': [x0, y0, x1 - x0, y1 - y0],  # Преобразуем в [x, y, w, h]
                    'best_conf': conf,
                    'best_cls': int(cls)
                })
            else:
                print(f"Неверный формат бокса: {box}")

    return boxes

def run_predict(input_path, model, score_threshold=0.5, iou_threshold=0.7):
    """
    Run prediction on all images and collect results.
    """
    predictions = []

    img_paths = [input_path]  # Здесь можно указать список путей к изображениям

    for img_path in img_paths:
        results = results_predict(img_path, model, score_threshold, iou_threshold)
        predictions.extend(results)

    return predictions


def main():
    model_path = 'yolov5x.pt'
    threshold = 0.5
    iou_threshold = 0.7
    annotations_file = 'annotations/instances_train2017.json'

    # Загружаем модель
    model = YOLO(model_path)

    # Загружаем аннотации COCO
    coco = COCO(annotations_file)

    # Запускаем предсказания и вычисляем mAP
    predictions = run_predict('val2017', model, score_threshold=threshold, iou_threshold=iou_threshold)

    # Вычисляем mAP
    mAP, ap_per_class = evaluate_mAP(predictions, coco, iou_threshold=0.5)

    print(f"mAP: {mAP}")
    for category_id, ap in ap_per_class.items():
        print(f"AP for class {category_id}: {ap}")


main()