import numpy as np 

def find_iou(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    x1 = max(x1A, x1B)
    y1 = max(y1A, y1B)
    x2 = min(x2A, x2B)
    y2 = min(y2A, y2B)

    area_intersect = (x2 - x1) * (y2 - y1)
    area_A = (x2A - x1A) * (y2A - y1A)
    area_B = (x2B - x1B) * (y2B - y1B)

    return area_intersect / (area_A + area_B - area_intersect)
    
def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    # Write code here
    boxes = np.asarray(boxes, dtype = np.float64)
    scores = np.asarray(scores, dtype = np.float64)
    
    order = np.argsort(-scores)
    # print(order)
    boxes = boxes[order]
    scores = scores[order]
    n = len(boxes)
        
    for i in range(n):
        if scores[i] == -1:
            continue 

        for j in range(i + 1, n):
            if find_iou(boxes[i], boxes[j]) >= iou_threshold:
                scores[j] = -1
                order[j] = -1

    res = []
    order = order.tolist()
    for x in order:
        if x != -1:
            res.append(x)
    return res