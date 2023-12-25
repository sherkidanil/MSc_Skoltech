from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def func(x):
    return np.power(np.exp(x) - np.cos(x),2)
    
points_num = 2*10**8

# now I defined the step 
step = points_num//size

# let's create linspace of x
space = np.linspace(0,3, points_num)

if rank != size:
    subspace = space[rank*step:(rank+1)*step]
else:
    subspace = space[rank*step:]

result = np.trapz(func(subspace), subspace)
result = np.array(result, dtype=np.float32)
global_result = None

if rank == 0:
    global_result = np.empty(1, dtype=np.float32)

comm.Reduce(result, global_result, op=MPI.SUM, root=0)

if rank==0:
    np.save('global_result.npy', global_result)
    print(global_result)

MPI.Finalize()
