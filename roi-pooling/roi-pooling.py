import math
import numpy as np
def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    res = []
    feature_map  = np.array(feature_map)
    for roi in rois:
        x1, y1, x2, y2 = tuple(roi)

        h = y2 - y1 
        w = x2 - x1 
        bin = [[0 for i in range(output_size)] for j in range(output_size)]
        for i in range(output_size):
            for j in range(output_size):
                h_start = y1 + (i * h) // output_size
                h_end = y1 + ((i + 1) * h) // output_size
                h_end += (h_end == h_start)

                w_start = x1 + (j * w) // output_size 
                w_end = x1 + ((j + 1) * w) // output_size
                w_end += (w_end == w_start)

                bin[i][j] = int(np.max(feature_map[h_start:h_end, w_start:w_end]))

        res.append(bin)
    return res

        

        
        