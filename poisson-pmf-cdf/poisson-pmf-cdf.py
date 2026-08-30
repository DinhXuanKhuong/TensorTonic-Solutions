import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    pfm = (math.exp(-lam) * lam**k) / math.factorial(k)
    cdf = sum([(math.exp(-lam) * lam**i) / math.factorial(i) for i in range(0, k +1)])

    return {"pmf" : pfm, "cdf" : cdf}