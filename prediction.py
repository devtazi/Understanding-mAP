from utils import calculate_iou
import random
import config

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

def simulate_prediction(iteration):
    prediction_results = {}
    total_real_objects = {}

    for img_idx in range(config.NOMBRES_IMAGES_SIMULEES):

        data = next(iteration)
        objects = data['objects']

        for obj_idx in range(len(objects['bbox'])):
            gt_box = objects['bbox'][obj_idx]
            category = objects['category'][obj_idx]

            total_real_objects[category] += 1

            # Simulation d'une prédiction
            pred_box, score = generate_fake_prediction(
                gt_box, data['width'], data['height']
            )

            iou = calculate_iou(gt_box, pred_box)

            is_tp = 1 if iou >= config.SEUIL_IOU else 0
            prediction_results[category].append((score, is_tp))

    return prediction_results, total_real_objects