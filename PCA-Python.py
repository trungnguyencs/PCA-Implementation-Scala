import os
import psutil
import time
import numpy as np
import pandas as pd
import glob

def clean_output(outp_name):
    if os.path.exists(outp_name):
      os.remove(outp_name)

def trim_file_name(f_name):
    f_name = f_name.split('.')[1].split('/')[-1]
    print(f_name)
    return f_name

def extract_file_name(f_name):
    dim = int(f_name.split('_')[0])
    total_sample = int(f_name.split('_')[1])
    total_class = int(f_name.split('_')[2])
    reduce_dim = int(f_name.split('_')[3])
    return dim, total_sample, total_class, reduce_dim

def read_csv(f_name):
    data_stack = pd.read_csv('./data/' + f_name + '.csv', header = None)
    return data_stack.values

def compute_mean_vector(data_stack):
    """
    Geting the mean vector from the whole dataset
    """
    return np.average(data_stack, axis=1)

def compute_scatter_matrix(data_stack):
    """
    Computing the Scatter Matrix from the whole dataset
    """
    scatter_mat = np.zeros((dim,dim))
    for i in range(dim):
        scatter_mat += (data_stack[:,i].reshape(dim,1) - mean_vec).dot((data_stack[:,i].reshape(dim,1) - mean_vec).T)
    return scatter_mat

if __name__ == '__main__':

    outp_name = 'output_python.csv'
    clean_output(outp_name)

    f_list = glob.glob(os.path.join('./data', '*.csv'))
    
    cols = ['dim', 'total_sample', 'total_class', 'reduce_dim',\
     'load_data_runtime', 'total_runtime', 'memory_usage']
    output = pd.DataFrame(columns=cols)

    for f_name in f_list:
        start_time = time.time()
        process = psutil.Process(os.getpid()) 

        f_name = trim_file_name(f_name)
        dim, total_sample, total_class, reduce_dim = extract_file_name(f_name)
        print('Number of dimmensions: ' + str(dim))
        print('Number of samples in each class: ' + str(total_sample))
        print('Number of classes: ' + str(total_class))
        print('Number of reduced dimmensions: ' + str(reduce_dim))

        data_stack = read_csv(f_name)
        print('All data shape: ' + str(data_stack.shape[1]) + ' rows, ' + str(data_stack.shape[0]) + ' columns')

        data_time = time.time()
        
        # Geting the mean vector
        mean_vec = compute_mean_vector(data_stack)
        print('Mean vector:\n' + str(mean_vec))

        # Computing the Scatter Matrix
        scatter_mat = compute_scatter_matrix(data_stack)
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

        load_data_runtime = (data_time - start_time)*1024.0
        total_runtime = (time.time() - start_time)*1024.0
        memory_usage = process.memory_info().rss/(1024.0*1024.0)
        print("\n--- Load data runtime: %.4f miliseconds ---" % (load_data_runtime))
        print("--- Total runtime: %.4f miliseconds ---" % (total_runtime))
        print("--- Memory usage: %.4f Mbytes ---\n" % (memory_usage)  )

        temp_df = pd.DataFrame([[dim, total_sample, total_class, reduce_dim, load_data_runtime, total_runtime, memory_usage]], columns=cols)
        output = output.append(temp_df, ignore_index=True)

    output.to_csv(outp_name)