import torch
from pycocotools.coco import COCO
import cv2
import os
import torchvision.ops as ops

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

annFile = 'annotations/instances_val2017.json'
coco = COCO(annFile)

#список изображений
imgIds = coco.getImgIds()

#выбираем первое изображение из списка
img_id = imgIds[0]

def compute_iou(box1, box2):

    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    #площадь пересечения
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    #площадь объединения
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y2_gt)

    union_area = box1_area + box2_area - inter_area

    #IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

#оценка с NMS
def evaluate_image_with_nms(img_id):

    img = coco.loadImgs(img_id)[0]
    img_path = 'val2017/000000022396.jpg'

    img_cv = cv2.imread(img_path)

    #из BGR в RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    model.eval()

    #отключаем вычисление градиентов
    with torch.no_grad():
        #предсказания YOLO
        results = model(img_rgb)

        predictions = results.pred[0]

        #список классов для меток
        class_names = model.names

        #реальные аннотации для этого изображения ground truth
        annIds = coco.getAnnIds(imgIds=img_id)  #получаем id аннотаций для изображения
        anns = coco.loadAnns(annIds)  #загружаем аннотации

        #преобразуем предсказания в нужный формат для NMS
        boxes = predictions[:, :4]  #координаты bounding boxes [x1,y1,x2,y2]
        scores = predictions[:, 4]  #уверенность предсказания
        labels = predictions[:, 5]  #класс

        #NMS
        nms_indices = ops.nms(boxes, scores, 0.4)  #0.4 - порог IoU для NMS
        nms_boxes = boxes[nms_indices]
        nms_scores = scores[nms_indices]
        nms_labels = labels[nms_indices]

        #словарь для хранения IoU для каждого объекта
        iou_results = {}

        #рисуем все аннотации (ground truth) с другим цветом
        for ann in anns:
            gt_box = ann['bbox']  #координаты [x, y, width, height]
            gt_x1, gt_y1, gt_width, gt_height = gt_box
            gt_x2 = gt_x1 + gt_width
            gt_y2 = gt_y1 + gt_height
            gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]  #преобразуем в формат [x1, y1, x2, y2]

            #рисуем ground truth bounding box красным цветом
            gt_x1, gt_y1, gt_x2, gt_y2 = map(int, gt_box)
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(img_cv, (gt_x1, gt_y1), (gt_x2, gt_y2), color, thickness)

            #метка с названием класса для ground truth
            text = f'GT: {ann["category_id"]}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = gt_x1
            text_y = gt_y1 - 10
            cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, color, font_thickness)

            #для каждого оставшегося предсказания находим наиболее подходящую аннотацию (по IoU)
        for i, pred in enumerate(nms_boxes):
            x1, y1, x2, y2 = map(int, pred)
            confidence = nms_scores[i].item()
            label = int(nms_labels[i].item())

            #рисуем bounding box для предсказания зеленым цветом
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)

            #метка с названием класса и уверенностью для предсказания
            text = f'{class_names[label]}: {confidence:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10
            cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, color, font_thickness)

            #IoU
            max_iou = 0
            best_gt = None
            for gt_ann in anns:
                gt_box = gt_ann['bbox']  #координаты [x, y, width, height]
                gt_x1, gt_y1, gt_width, gt_height = gt_box
                gt_x2 = gt_x1 + gt_width
                gt_y2 = gt_y1 + gt_height
                gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]  #преобразуем в формат [x1, y1, x2, y2]

                iou = compute_iou(pred.tolist(), gt_box)

                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt_ann

            iou_results[f"Pred {i + 1}"] = {"IoU": max_iou,
                                            "GT Category ID": best_gt["category_id"] if best_gt else None}

        for pred_id, result in iou_results.items():
            print(f"{pred_id}: IoU = {result['IoU']:.4f}, GT Category ID = {result['GT Category ID']}")

        cv2.imshow('Image with Bounding Boxes', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

evaluate_image_with_nms(img_id)