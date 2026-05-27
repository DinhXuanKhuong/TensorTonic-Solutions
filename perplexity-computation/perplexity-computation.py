import math
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    log_prob = 0
    n = len(actual_tokens)
    for i in range(n):
        prob = math.log(prob_distributions[i][actual_tokens[i]])
        log_prob += prob
    log_prob /= -n

    return math.exp(log_prob)