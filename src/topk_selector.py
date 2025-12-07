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
def _quickselect_desc(arr, k_index):
    left = 0
    right = arr.shape[0] - 1
    while True:
        if left == right:
            return
        pivot_index = (left + right) // 2
        pivot_index = _partition_desc(arr, left, right, pivot_index)
        if k_index == pivot_index:
            return
        elif k_index < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


@njit(fastmath=True)
def _sort_small_desc(arr, k):
    for i in range(k - 1):
        for j in range(i + 1, k):
            if arr[j, 1] > arr[i, 1]:
                _swap(arr, i, j)


def topk_desc(arr, k):

    n = arr.shape[0]

    if n == 0 or k == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if k >= n:
        out = arr.astype(np.float32, copy=True)
        idx = np.argsort(out[:, 1])[::-1]
        return out[idx]


    tmp = arr.astype(np.float32, copy=True)


    _quickselect_desc(tmp, k - 1)

    out = tmp[:k].copy()

   
    if k <= 200:
        _sort_small_desc(out, k)
        return out 

    idx = np.argsort(out[:, 1])[::-1]
    return out[idx]
