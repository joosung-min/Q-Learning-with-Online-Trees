from math import log

def mean(xs):
    return sum(xs) / (len(xs)*1.0)

def sd(xs): 
    n = len(xs) *1.0
    mu = sum(xs) / n
    return((sum(list(map(lambda x: (x-mu)*(x-mu),xs))) / (n-1) )**(1/2))

def argmax(x):
    """
    returns the index of the element is a list which corresponds to the maximum
    """
    return x.index(max(x))

def log2(x):
    """
    log base 2
    """
    return log(x) / log(2)