import os
import psutil
import time

start_time = time.time()
process = psutil.Process(os.getpid())

if __name__ == '__main__':
    
    import numpy as np
    import pandas as pd

    fname = '4_100_2_2.csv'
    data_stack = pd.read_csv(fname, header = None)
    data_stack = data_stack.values

    dim = int(fname.split('.')[0].split('_')[0])
    total_sample = int(fname.split('.')[0].split('_')[1])
    total_class = int(fname.split('.')[0].split('_')[2])
    reduce_dim = int(fname.split('.')[0].split('_')[3])

    print('Number of dimmensions: ' + str(dim))
    print('Number of samples in each class: ' + str(total_sample))
    print('Number of classes: ' + str(total_class))
    print('Number of reduced dimmensions: ' + str(reduce_dim))
    print('All data shape: ' + str(data_stack.shape[1]) + ' rows, ' + str(data_stack.shape[0]) + ' columns')

    # Geting the mean vector
    mean_vec = np.average(data_stack, axis=1)
    print('Mean vector:\n' + str(mean_vec))

    # Computing the Scatter Matrix
    scatter_mat = np.zeros((dim,dim))
    for i in range(dim):
        scatter_mat += (data_stack[:,i].reshape(dim,1) - mean_vec).dot((data_stack[:,i].reshape(dim,1) - mean_vec).T)
    print('Scatter Matrix:\n' + str(scatter_mat))

    # Computing the Eigenvalues and Eigenvectors
    eig_val, eig_mat = np.linalg.eig(scatter_mat)
    print('Eigenvalues:\n' + str(eig_val))
    print('Eigenvectors:\n' + str(eig_mat))

    # Sort the Eigenvalues in decreasing order
    np.argsort(np.abs(eig_val))[::-1]

    # Reduce (reduce_dim) dimensions: Take only (dim-reduce_dim) scatter vectors to put in the new eigen matrix
    new_eig_mat = []
    for i in range(0, dim-reduce_dim):
      new_eig_mat.append(eig_mat[i])
    new_eig_mat = np.asarray(new_eig_mat) 
    print('New eigenvectors:\n' + str(new_eig_mat))

    # Project all samples to the new subspace:
    proj_data = new_eig_mat.dot(data_stack)
    print('Projected data shape:\n' + str(proj_data.shape[1]) + ' rows, ' + str(proj_data.shape[0]) + ' columns')

print("--- Runtime: %s seconds ---" % (time.time() - start_time))
print("--- Memory usage: %s Mbytes ---" % (process.memory_info().rss/1000000.0))  


