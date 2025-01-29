import os
import json
import time
import torch
import numpy as np
import torchvision.ops as tv_ops

from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

################################################################################
# HOOK
################################################################################
class SaveIO:
    """Hook для сохранения input/output конкретного слоя модели."""
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

################################################################################
# ЗАГРУЗКА МОДЕЛИ С ПОДКЛЮЧЕНИЕМ HOOK
################################################################################
def load_and_prepare_model(model_path='yolov5x.pt', device='cuda'):
    """
    Загружает модель YOLO (Ultralytics), ищет слой Detect и подключает к нему хук,
    позволяющий после forward получить «сырые» выходы (logits/активации).
    Переносит модель на нужное устройство (GPU/CPU).
    """
    # Загружаем модель
    model = YOLO(model_path)
    model.to(device)  # перенос на CUDA, если доступно

    detect_layer = None
    detect_hook = SaveIO()

    # Регистрируем хук на слой Detect
    for module in model.model.modules():
        if isinstance(module, Detect):
            module.register_forward_hook(detect_hook)
            detect_layer = module
            break

    return model, detect_layer, detect_hook

################################################################################
# ПАРАМЕТРЫ, ЗАГРУЗКА COCO
################################################################################
ann_file = r'annotations/instances_val2017.json'
coco = COCO(ann_file)

img_ids = coco.getImgIds()

CONF_THRES = 0.5
IOU_THRES_NMS = 0.7
IOU_THRES_GT = 0.5  # порог IoU для TP

################################################################################
# ФУНКЦИИ ДЛЯ IoU И mAP
################################################################################
def compute_iou(box1, box2):
    """Вычисляет IoU на CPU (формат [x1,y1,x2,y2])."""
    x1_p, y1_p, x2_p, y2_p = box1
    x1_g, y1_g, x2_g, y2_g = box2

    inter_x1 = max(x1_p, x1_g)
    inter_y1 = max(y1_p, y1_g)
    inter_x2 = min(x2_p, x2_g)
    inter_y2 = min(y2_p, y2_g)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_pred = (x2_p - x1_p) * (y2_p - y1_p)
    area_gt = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = area_pred + area_gt - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def compute_map(all_results):
    """
    Подсчёт mAP по упрощённой схеме через TP/FP.
    all_results[label] = [0/1, 0/1, ...].
    Возвращает (aps, mAP).
    """
    aps = {}
    for label, tp_fp_list in all_results.items():
        tp = np.array(tp_fp_list, dtype=float)
        fp = 1.0 - tp

        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)

        total_gt_for_class = np.sum(tp)  # упрощённо
        if total_gt_for_class == 0:
            aps[label] = 0.0
            continue

        recall = cumsum_tp / total_gt_for_class
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-16)

        recall_levels = np.linspace(0, 1, 101)
        prec_interp = []
        for rl in recall_levels:
            p_candidates = precision[recall >= rl]
            p = max(p_candidates) if len(p_candidates) > 0 else 0
            prec_interp.append(p)

        ap = np.mean(prec_interp)
        aps[label] = ap

    if len(aps) == 0:
        return aps, 0.0
    mAP = np.mean(list(aps.values()))
    return aps, mAP

################################################################################
# ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ (С HOOK), NMS НА GPU, ВОЗВРАТ АКТИВАЦИЙ
################################################################################
def results_predict_with_hook(img_path, model, detect_layer, detect_hook,
                              conf_thres=0.5, iou_thres=0.7, device='cuda'):
    """
    Делает forward модели (на GPU), вытягивает сырые выходы из detect_hook.output.
    Затем вручную применяет фильтрацию и NMS (на GPU),
    возвращая [{'bbox': [...], 'score': ..., 'activations': [...]}].
    """
    # Запускаем модель (детекция). Это «прогон» на GPU, если device='cuda'.
    _ = model.predict(img_path, device=device)  # ultralytics>=8: model.predict

    raw_out = detect_hook.output[0]  # Тензор формы [1, c, nb], либо [batch, c, nb]
    if raw_out is None:
        return []

    # Убедимся, что сырой выход на GPU
    raw_out = raw_out.to(device)

    # Предположим формат [1, 85, N], где:
    #  - первые 4: x1,y1,x2,y2
    #  - 5-й: conf
    #  - 6.. : class_probs
    # Переставим оси, чтобы получить [N, 85].
    if raw_out.dim() == 3:
        raw_out = raw_out.permute(2, 1, 0).squeeze(-1)  # -> [N, 85]
    else:
        # fallback, если вдруг [N, 85] уже
        pass

    if raw_out.shape[0] == 0:
        return []

    boxes_xyxy = raw_out[:, 0:4]    # [N, 4]
    conf_scores = raw_out[:, 4]     # [N]
    class_probs = raw_out[:, 5:]    # [N, num_classes]

    # Фильтруем по conf_thres на GPU
    mask = conf_scores >= conf_thres
    boxes_xyxy = boxes_xyxy[mask]
    conf_scores = conf_scores[mask]
    class_probs = class_probs[mask]

    if boxes_xyxy.size(0) == 0:
        return []

    # NMS на GPU
    keep_idx = tv_ops.nms(boxes_xyxy, conf_scores, iou_thres)
    final_boxes = boxes_xyxy[keep_idx]
    final_scores = conf_scores[keep_idx]
    final_probs = class_probs[keep_idx]

    # Переводим на CPU для удобства формирования python-объектов
    final_boxes = final_boxes.detach().cpu().numpy()
    final_scores = final_scores.detach().cpu().numpy()
    final_probs = final_probs.detach().cpu().numpy()

    # Сбор результатов
    out_list = []
    for i in range(final_boxes.shape[0]):
        box = final_boxes[i].tolist()  # [x1, y1, x2, y2]
        score_i = float(final_scores[i])
        probs_i = final_probs[i].tolist()  # список из num_classes

        out_list.append({
            'bbox': box,
            'score': score_i,
            'activations': probs_i
        })

    return out_list

################################################################################
# ФУНКЦИЯ: БЕРЁМ TOP-2 ПО АКТИВАЦИЯМ (На каждый бокс 2 предсказания)
################################################################################
def predict_with_top2(img_path, model, detect_layer, detect_hook, device='cuda'):
    """
    1) Использует results_predict_with_hook(...) для получения боксов и всего вектора активаций.
    2) Для каждого бокса берёт top-2 классов => формирует 2 предикта.
    3) Складывает в список, сортирует по score (убывание).
    """
    predictions = results_predict_with_hook(
        img_path, model, detect_layer, detect_hook,
        conf_thres=CONF_THRES, iou_thres=IOU_THRES_NMS, device=device
    )
    final_list = []
    for pred in predictions:
        box = pred['bbox']
        activations = pred['activations']  # список float по всем классам
        tensor_acts = torch.tensor(activations)

        top_vals, top_inds = tensor_acts.topk(2)  # берём 2 максимума
        for i in range(2):
            cls_i = int(top_inds[i].item())
            score_i = float(top_vals[i].item())

            final_list.append({
                'box': box,
                'score': score_i,
                'cls': cls_i
            })

    # Сортируем по score
    final_list = sorted(final_list, key=lambda x: x['score'], reverse=True)
    return final_list

################################################################################
# ОСНОВНАЯ ОЦЕНКА NA COCO (TP/FP, mAP)
################################################################################
def evaluate_model_top2_on_coco(model, detect_layer, detect_hook, coco, device='cuda'):
    """
    Пробегается по всем изображениям, для каждого берёт top-2 по каждому боксу,
    сверяет с GT, набирает TP/FP, вычисляет mAP.
    """
    all_results = {}
    img_ids = coco.getImgIds()

    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join('val2017', img_info['file_name'])

        preds_top2 = predict_with_top2(img_path, model, detect_layer, detect_hook, device=device)

        # Считываем GT
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_list = []
        for ann in anns:
            cat_id = ann['category_id']
            x, y, w, h = ann['bbox']
            gt_box = [x, y, x + w, y + h]
            gt_list.append((cat_id, gt_box))

        used_gt = set()

        for pred in preds_top2:
            pred_cls = pred['cls']
            pred_score = pred['score']
            pred_box = pred['box']

            # Ищем лучший IoU для того же класса
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (gt_cls, gt_box) in enumerate(gt_list):
                if gt_cls != pred_cls:
                    continue
                if gt_idx in used_gt:
                    continue
                iou_val = compute_iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx

            if best_iou >= IOU_THRES_GT and best_gt_idx >= 0:
                is_tp = 1
                used_gt.add(best_gt_idx)
            else:
                is_tp = 0

            if pred_cls not in all_results:
                all_results[pred_cls] = []
            all_results[pred_cls].append(is_tp)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1} / {len(img_ids)} images")

    aps, mAP = compute_map(all_results)
    return aps, mAP

################################################################################
# MAIN
################################################################################
def main():
    # Определяем устройство (GPU если доступно)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    start_time = time.time()

    model, detect_layer, detect_hook = load_and_prepare_model('yolov5x.pt', device=device)

    aps, mAP = evaluate_model_top2_on_coco(model, detect_layer, detect_hook, coco, device=device)

    print(f"\nMean Average Precision (mAP) with top-2 classes per box: {mAP:.4f}")
    for label, ap in aps.items():
        print(f"AP for class {label}: {ap:.4f}")

    elapsed = time.time() - start_time
    print(f"\nDone. Elapsed time: {elapsed:.2f} sec.")

if __name__ == "__main__":
    main()
