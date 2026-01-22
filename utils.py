import random 
import numpy as np

def calculate_area_of_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)

def calculate_iou(ground_truth_bbox, prediction_bbox):
    area_of_overlap = calculate_area_of_overlap(ground_truth_bbox, prediction_bbox)
    area1 = ground_truth_bbox[2] * ground_truth_bbox[3]
    area2 = ground_truth_bbox[2] * ground_truth_bbox[3]
    area_of_union = area1 + area2 - area_of_overlap
    return area_of_overlap / area_of_union if area_of_union != 0 else 0

def generate_fake_prediction(ground_truth_bbox, image_w, image_h, ratio=0.3):
    x, y, w, h = ground_truth_bbox

    dx = w * random.uniform(-ratio, ratio)
    dy = h * random.uniform(-ratio, ratio) 
    dw = w * random.uniform(-ratio, ratio)
    dh = h * random.uniform(-ratio, ratio)

    # On travaille avec les coins pour la suite 
    # (x1, y1) = coin supérieur gauche
    # (x2, y2) = coin inférieur droit
    x1 = x + dx
    y1 = y + dy
    x2 = x1 + max(1, w + dw)
    y2 = y1 + max(1, h + dh) 

    # On s'assure que la nouvelle bbox reste dans l'image
    x1 = max(0, min(x1, image_w))
    y1 = max(0, min(y1, image_h))
    x2 = max(0, min(x2, image_w))
    y2 = max(0, min(y2, image_h))

    # On recalcule w et h
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return [0, 0, 0, 0],0
    
    # On simule un score de confiance entre 0.1 et 1.0
    score = random.uniform(0.1, 1.0)

    return [x1, y1, w, h], score

def compute_average_precision(recall, precision):

    # Fermeture de la courbe
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))

    # Lissage de la courbe de précision
    for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calcul de l'aire sous la courbe
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap