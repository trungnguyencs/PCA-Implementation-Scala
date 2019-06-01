import os
import psutil
import time
import numpy as np
import pandas as pd
import glob

def clean_output(outp_name):
    """
    Delete the output csv file if exits
    """       
    if os.path.exists(outp_name):
      os.remove(outp_name)

def trim_file_name(f_name):
    """
    Get the part containing dim, total_sample, total_class, reduced_dim from csv file name
    """    
    f_name = f_name.split('.')[1].split('/')[-1]
    print(f_name)
    return f_name

def extract_file_name(f_name):
    """
    Get the dim, total_sample, total_class, reduced_dim information from csv file name
    """
    dim = int(f_name.split('_')[0])
    total_sample = int(f_name.split('_')[1])
    total_class = int(f_name.split('_')[2])
    reduced_dim = int(f_name.split('_')[3])
    return dim, total_sample, total_class, reduced_dim

def read_csv(f_name):
    """
    Return the dataset in numpy array format from reading csv file
    """
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

def compute_eig(scatter_mat):
    """
    Computing the eigenvalues and eigenvectors from the scatter matrix
    """
    eig_val, eig_mat = np.linalg.eig(scatter_mat)
    return eig_val, eig_mat

def reduce_dim(eig_mat, dim, reduced_dim):
    """
    Reduce (reduced_dim) dimensions: Take only (dim-reduced_dim) scatter vectors to put in the new eigen matrix
    """
    new_eig_mat = []
    for i in range(0, dim-reduced_dim):
      new_eig_mat.append(eig_mat[i])
    new_eig_mat = np.asarray(new_eig_mat)
    return new_eig_mat

def project_data(new_eig_mat, data_stack):
    """
    Project the original dataset to the new subspace using the remaining eigenvectors
    """
    return new_eig_mat.dot(data_stack)


if __name__ == '__main__':

    outp_name = 'output_python.csv'
    clean_output(outp_name)

    f_list = glob.glob(os.path.join('./data', '*.csv'))
    
    cols = ['dim', 'total_sample', 'total_class', 'reduced_dim',\
     'load_data_runtime', 'total_runtime', 'memory_usage']
    output = pd.DataFrame(columns=cols)

    for f_name in f_list:
        # Set the time time stamp and process id for memory measurement
        start_time = time.time()
        process = psutil.Process(os.getpid()) 

        # Trim file name and get dim, total_sample, total_class, reduced_dim from there
        f_name = trim_file_name(f_name)
        dim, total_sample, total_class, reduced_dim = extract_file_name(f_name)
        print('Number of dimmensions: ' + str(dim))
        print('Number of samples in each class: ' + str(total_sample))
        print('Number of classes: ' + str(total_class))
        print('Number of reduced dimmensions: ' + str(reduced_dim))

        # Get the dataset in numpy array format from reading csv file
        data_stack = read_csv(f_name)
        print('All data shape: ' + str(data_stack.shape[1]) + ' rows, ' + str(data_stack.shape[0]) + ' columns')

        # Get the timestamp right after finishing reading the csv file
        data_time = time.time()
        
        # Geting the mean vector
        mean_vec = compute_mean_vector(data_stack)
        print('Mean vector:\n' + str(mean_vec))

        # Computing the scatter matrix
        scatter_mat = compute_scatter_matrix(data_stack)
        print('Scatter Matrix:\n' + str(scatter_mat))

        # Computing the eigenvalues and eigenvectors
        eig_val, eig_mat = compute_eig(scatter_mat)
        print('Eigenvalues:\n' + str(eig_val))
        print('Eigenvectors:\n' + str(eig_mat))

        # Sort the Eigenvalues in decreasing order
        np.argsort(np.abs(eig_val))[::-1]

        # Reduce (reduced_dim) dimensions: Take only (dim-reduced_dim) scatter vectors to put in the new eigen matrix
        new_eig_mat = reduce_dim(eig_mat, dim, reduced_dim)
        print('New eigenvectors:\n' + str(new_eig_mat))

        # Project all samples to the new subspace:
        proj_data = new_eig_mat.dot(data_stack)
        print('Projected data shape:\n' + str(proj_data.shape[1]) + ' rows, ' + str(proj_data.shape[0]) + ' columns')

        # Calculate load data runtime, total runtime and memory usage
        load_data_runtime = (data_time - start_time)*1024.0
        total_runtime = (time.time() - start_time)*1024.0
        memory_usage = process.memory_info().rss/(1024.0*1024.0)
        print("\n--- Load data runtime: %.4f miliseconds ---" % (load_data_runtime))
        print("--- Total runtime: %.4f miliseconds ---" % (total_runtime))
        print("--- Memory usage: %.4f Mbytes ---\n" % (memory_usage)  )

        # Create a pandas data frame of output
        temp_df = pd.DataFrame([[dim, total_sample, total_class, reduced_dim, load_data_runtime, total_runtime, memory_usage]], columns=cols)
        output = output.append(temp_df, ignore_index=True)

    # Write output to csv file
    output.to_csv(outp_name)