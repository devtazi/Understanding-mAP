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

def compute_average_precision(recall, precision):

    # Fermeture de la courbe
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Lissage de la courbe de précision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calcul de l'aire sous la courbe
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calculate_map(prediction_results, total_real_objects):
    aps = {}

    for category, preds in prediction_results.items():

        # Trier par score décroissant
        preds.sort(key=lambda x: x[0], reverse=True)

        tp_cum = []
        fp_cum = []

        tp_sum = 0
        fp_sum = 0

        for _, is_tp in preds:
            if is_tp:
                tp_sum += 1
            else:
                fp_sum += 1

            tp_cum.append(tp_sum)
            fp_cum.append(fp_sum)

        tp_cum = np.array(tp_cum)
        fp_cum = np.array(fp_cum)

        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        recalls = tp_cum / total_real_objects[category]

        ap = compute_average_precision(recalls, precisions)
        aps[category] = ap

    return np.mean(list(aps.values()))