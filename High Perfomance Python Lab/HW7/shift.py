import cv2
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

PATH = 'tesla_2.jpeg'

def shifting(img):
    img = img.copy()
    n_iter = img.shape[0]
    imgs = []
    for _ in range(n_iter):
        img = np.roll(img, shift=1, axis=0)
        imgs.append(img.copy())
    return np.array(imgs)

if rank == 0:
    img = cv2.imread(PATH, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width_part = img.shape[1] // size
    imgs_stacked = [img[:, i * width_part:(i + 1) * width_part] for i in range(size)]
else:
    imgs_stacked = None

img_part = comm.scatter(imgs_stacked, root=0)
img_parts_shifted = shifting(img_part)

n_iter, height, width_part, n_channels = img_parts_shifted.shape

if rank == 0:
    img_parts_shifted_stacked = np.empty((size, n_iter, height, width_part, n_channels), dtype=np.uint8) 
else:
    img_parts_shifted_stacked =  None

comm.Gatherv(img_parts_shifted, img_parts_shifted_stacked, root=0)

if rank == 0:
    img_parts_shifted_stacked = np.concatenate(list(img_parts_shifted_stacked), axis=2)
    np.save('result.npy', img_parts_shifted_stacked)

MPI.Finalize()
