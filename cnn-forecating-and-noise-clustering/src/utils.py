import numpy as np
import heapq

def null_percentages(col):
    col_is_null = col.isnull()
    return len(col_is_null[col.isnull()==True])/len(col_is_null)

def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]
	
def interpolate_nans(y):
    y = y.values
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y