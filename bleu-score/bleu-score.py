import math
def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    # Write code here
    c = len(candidate)
    r = len(reference)
    if c == 0:
        return 0.0
        
    brevity_penalty = 1 if c >= r else math.exp(1 - r / c)
    modified_pre = []

    for n_gram in range(max_n):
        candidate_dict = dict()
        reference_dict = dict()
        p = 0
        print(f"n_gram: {n_gram + 1}")
        for i in range(len(reference) - n_gram):
            ref = ' '.join(reference[i: i+ n_gram + 1])
            reference_dict[ref] = reference_dict.get(ref, 0) + 1
            
        for i in range(len(candidate) - n_gram):
            candi = ' '.join(candidate[i: i+ n_gram + 1])
            candidate_dict[candi] = candidate_dict.get(candi, 0) + 1
        
        for k in candidate_dict:
            p += min(candidate_dict[k], reference_dict.get(k, 0))
        
        print(f"candidate: {candidate_dict}")
        print(f"reference: {reference_dict}")
        
        p /= (len(candidate) - n_gram)
        print(p)
        if p == 0:
            return 0.0
        modified_pre.append(p)
    
    bleu = brevity_penalty * math.exp(sum([math.log(pre) for pre in modified_pre]) / max_n)
    return bleu        
        

    