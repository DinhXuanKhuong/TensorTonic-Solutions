from datetime import datetime 
from operator import itemgetter
def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    for m in models:
        m["latency"] *= -1
    res = sorted(models, key = itemgetter("accuracy", "latency", "timestamp"), reverse = True)
    return res[0]['name']