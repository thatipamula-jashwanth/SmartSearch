import numpy as np
from numba import njit

@njit(fastmath=True)
def _swap(arr, i, j):
    t0 = arr[i, 0]
    t1 = arr[i, 1]
    arr[i, 0] = arr[j, 0]
    arr[i, 1] = arr[j, 1]
    arr[j, 0] = t0
    arr[j, 1] = t1


@njit(fastmath=True)
def _partition_desc(arr, left, right, pivot_index):
    pivot = arr[pivot_index, 1]
    _swap(arr, pivot_index, right)
    store = left
    for i in range(left, right):
        if arr[i, 1] > pivot:    
            _swap(arr, store, i)
            store += 1
    _swap(arr, store, right)
    return store


@njit(fastmath=True)
def _quickselect_desc(arr, k):
    left = 0
    right = arr.shape[0] - 1
    while True:
        if left == right:
            return
        pivot_index = (left + right) // 2
        pivot_index = _partition_desc(arr, left, right, pivot_index)
        if k == pivot_index:
            return
        elif k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


@njit(fastmath=True)
def topk_desc(arr, k):
   
    n = arr.shape[0]

    # Edge cases
    if n == 0 or k == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if k >= n:
        out = arr.copy()
       
        idx = np.argsort(out[:, 1])[::-1]
        return out[idx].astype(np.float32)

   
    tmp = arr.copy()
    _quickselect_desc(tmp, k)

  
    out = tmp[:k].copy()

 
    if k <= 200:
       
        for i in range(k - 1):
            for j in range(i + 1, k):
                if out[j, 1] > out[i, 1]:
                    _swap(out, i, j)
    else:
        idx = np.argsort(out[:, 1])[::-1]
        out = out[idx]

    return out.astype(np.float32)
