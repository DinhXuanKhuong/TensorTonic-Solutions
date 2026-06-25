def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b
    
    x1, y1 = max(x1a, x1b), max(y1a, y1b)
    x2, y2 = min(x2a, x2b), min(y2a, y2b)
    print(x1, y1, x2, y2)
    intersection = (x2 - x1) * (y2 - y1) * (x2 > x1) * (y2 > y1)
    print(intersection)

    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - intersection

    return intersection / union