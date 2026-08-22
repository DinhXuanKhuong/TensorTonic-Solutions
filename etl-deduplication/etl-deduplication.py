def count_none(record):
    """
    record : dict
    """
    cnt = 0

    for k in record:
        cnt += (record[k] is None)
    return cnt
def is_same_key(keys, record1, record2):
    res = True
    for key in keys:
        if record1[key] != record2[key]:
            res = False
            break 
    return res

def put_in(res, record, keys, strategy):
    x = res.copy()
    flag = 0
    for i,r in enumerate(x):
        if is_same_key(keys, r, record) == True:
            if strategy == "most_complete":
                diff = count_none(record) - count_none(r)
                if diff < 0:
                    x[i] = record 
                    flag = 1
                elif diff == 0:
                    flag = 1
                    continue
            elif strategy == "first":
                flag = 1
                continue 
            elif strategy == "last":
                x[i] = record
                flag = 1
        
    if flag == 0:
        x.append(record)
    return x
        
    
def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    res = []
    for record in records:
        res = put_in(res, record, key_columns, strategy)
    return res
        