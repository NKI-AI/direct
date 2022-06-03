#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as cnp
from libc.math cimport abs, cos, pi, sin
from libc.stdlib cimport RAND_MAX, rand, srand

cnp.import_array()


cdef double random_uniform() nogil:
    """Produces a random number in (0, 1)."""
    cdef double r = rand()
    return r / RAND_MAX

cdef int randint(int upper) nogil:
    """Produces a random integer in {0, 1, ..., upper-1}."""
    return int(random_uniform() * (upper))

cdef inline Py_ssize_t fmax(Py_ssize_t one, Py_ssize_t two) nogil:
    """Max(a, b)."""
    return one if one > two else two

cdef inline Py_ssize_t fmin(Py_ssize_t one, Py_ssize_t two) nogil:
    """Min(a, b)."""
    return one if one < two else two

def poisson(
    int nx,
    int ny,
    int max_attempts,
    cnp.ndarray[cnp.int_t, ndim=2, mode='c'] mask,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] radius_x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] radius_y,
    cnp.ndarray[cnp.int_t, ndim=1] calib,
    int seed
):
    # Add calibration region
    cdef int x, y, num_actives, i, k
    cdef float rx, ry, v, t, qx, qy

    cdef Py_ssize_t startx, endx, starty, endy, px, py

    # initialize active list
    cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] pxs = np.empty(nx * ny, dtype=int)
    cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] pys = np.empty(nx * ny, dtype=int)

    with nogil:
        for x in range(int(nx / 2 - calib[0] / 2), int(nx / 2 + calib[0] / 2)):
            for y in range(int(ny / 2 - calib[1] / 2), int(ny / 2 + calib[1] / 2)):
                mask[x, y] = 1

        srand(seed)

        pxs[0] = randint(nx)
        pys[0] = randint(ny)

        num_actives = 1

        while num_actives > 0:
            i = randint(num_actives)
            px = pxs[i]
            py = pys[i]

            rx = radius_x[px, py]
            ry = radius_y[px, py]
            # Attempt to generate point
            done = False
            k = 0
            while not done and k < max_attempts:
                # Generate point randomly from r and 2 * r
                v = (random_uniform() * 3 + 1) ** 0.5
                t = 2 * pi * random_uniform()
                qx = px + v * rx * cos(t)
                qy = py + v * ry * sin(t)

                # Reject if outside grid or close to other points
                if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                    startx = fmax(int(qx - rx), 0)
                    endx = fmin(int(qx + rx + 1), nx)
                    starty = fmax(int(qy - ry), 0)
                    endy = fmin(int(qy + ry + 1), ny)

                    done = True
                    for x in range(startx, endx):
                        for y in range(starty, endy):
                            if (mask[x, y] == 1
                                and (((qx - x) / radius_x[y, x]) ** 2 +
                                     ((qy - y) / (radius_y[y, x])) ** 2 < 1)):
                                done = False
                                break

                k += 1

            # Add point if done else remove from active list
            if done:
                pxs[num_actives] = int(qx)
                pys[num_actives] = int(qy)
                mask[pxs[num_actives], pys[num_actives]] = 1
                num_actives += 1
            else:
                pxs[i] = pxs[num_actives - 1]
                pys[i] = pys[num_actives - 1]
                num_actives -= 1
