import numpy as np
import time

width = 1024
height = 1024
MAX_ITERS = 200
min_x = -2.0
max_x = 0.6
min_y = -1.5
max_y = 1.5

def mandelbrot_kernel(c):
    z = c
    for i in range(MAX_ITERS):
        z = z * z + c
        if np.linalg.norm(z) > 4:
            return i
    return MAX_ITERS


def compute_mandelbrot():

    t = np.zeros((height, width))

    dx = (max_x - min_x) / width
    dy = (max_y - min_y) / height

    y = min_y
    for row in range(height):
        x = min_x
        for col in range(width):
            c = x + 1j*y
            t[row, col] = mandelbrot_kernel(c)
            x += dx
        y += dy
    return t

if __name__ == '__main__':
    start = time.time()
    compute_mandelbrot()
    end = time.time()
    print(f'Python implementation: {(end - start)*1000} ms')