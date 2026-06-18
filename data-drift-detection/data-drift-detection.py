
def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    # pass
    tvd = 0
    s_r = sum(reference_counts)
    s_p = sum(production_counts)
    for i in range(len(reference_counts)):
        tvd += abs(reference_counts[i]/s_r - production_counts[i]/s_p)
    tvd /= 2
    res = {"score" : tvd , 
          "drift_detected": tvd > threshold}
    return res