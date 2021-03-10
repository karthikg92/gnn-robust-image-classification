import numpy as np

def random_coord(dims=[28,28], N=400):
    coord = [ [i,j] for i in range(dims[0]) for j in range(dims[1]) ]
    perm = np.random.permutation(coord)
    return perm[:N]

def compute_f(I, index):
    f=[]
    for i,j in index:
        f.append(I[:,i,j])
    f_array=np.concatenate(f, axis=1)
    f_extend=np.expand_dims(f_array, axis=2)
    return f_extend

def top_k(row, k=8):
    r=np.array(row)
    index=r.argsort()[-k]
    cutoff=r[index]
    return [0 if r<cutoff else r  for r in row]

def compute_A(k, index):
    N=len(index)
    A=[]
    for i in range(N):
        row = [ np.linalg.norm(np.array(index[i]) - np.array(index[j]) ) for j in range(N)]
        A.append(top_k(row, k))
    A=np.array(A)
    return A/np.max(A)

def image2graph_setup( N, k, dims=[28,28]):
    index=random_coord(dims, N)
    A=compute_A(k, index)
    return index, A

''' Format of usage 
index, A = image2graph_setup(N=5, k=3)
f = compute_f(Img, index) '''
