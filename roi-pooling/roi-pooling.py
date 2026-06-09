import math
import numpy as np
def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    res = []
    # feature_map = np.array(feature_map)
    for roi in rois:
        x1, y1, x2, y2 = tuple(roi)
        roi_h = y2 - y1
        roi_w = x2 - x1
        out_roi = [[0 for i in range(output_size)] for i in range(output_size)]
        for i in range(output_size):
            for j in range(output_size):
                h_start = y1 +  (i * roi_h) // output_size
                h_end = y1 + ((i + 1) * roi_h) // output_size
                w_start = x1 + (j * roi_w) // output_size
                w_end = x1 + ((j + 1) * roi_w)// output_size
                
                h_end = h_end + (h_end == h_start)
                w_end = w_end + (w_end == w_start)

                pool = [row[w_start:w_end] for row in feature_map[h_start:h_end]]

                out_roi[i][j] = max(max(row) for row in pool)
                # print(out_roi)
        res.append(out_roi)
    return res
                
        