import numpy as np
from numba import jit

@jit(nopython=True)
def boxes_iou3d_numba(boxes1, boxes2):
    N, M = boxes1.shape[0], boxes2.shape[0]
    iou = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            volume1 = boxes1[i, 3] * boxes1[i, 4] * boxes1[i, 5]
            volume2 = boxes2[j, 3] * boxes2[j, 4] * boxes2[j, 5]

            minx = max(boxes1[i, 0] - boxes1[i, 3] / 2, boxes2[j, 0] - boxes2[j, 3] / 2)
            maxx = min(boxes1[i, 0] + boxes1[i, 3] / 2, boxes2[j, 0] + boxes2[j, 3] / 2)
            miny = max(boxes1[i, 1] - boxes1[i, 4] / 2, boxes2[j, 1] - boxes2[j, 4] / 2)
            maxy = min(boxes1[i, 1] + boxes1[i, 4] / 2, boxes2[j, 1] + boxes2[j, 4] / 2)
            minz = max(boxes1[i, 2] - boxes1[i, 5] / 2, boxes2[j, 2] - boxes2[j, 5] / 2)
            maxz = min(boxes1[i, 2] + boxes1[i, 5] / 2, boxes2[j, 2] + boxes2[j, 5] / 2)

            overlap = max(0, maxx - minx) * max(0, maxy - miny) * max(0, maxz - minz)
            iou[i, j] = overlap / (volume1 + volume2 - overlap)

    return iou