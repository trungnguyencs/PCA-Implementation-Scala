import numpy as np

# Generate data input
np.random.seed(2019)
dim = 4
total_sample_lst = []
for i in range(10):
	total_sample_lst.append(50*2**i)
total_class = 2
reduce_dim = 2

for total_sample in total_sample_lst:
	data = []
	for i in range(total_class):
	  mean_vec = np.ones(dim)*i
	  cov_mat = np.identity(dim)
	  class_sample = np.random.multivariate_normal(mean_vec, cov_mat, total_sample).T
	  data.append(class_sample)
	data = np.asarray(data)

	# Ignoring class label
	data_stack = np.reshape(data, (dim, total_class*total_sample))
	np.savetxt('./data/' + str(dim) + '_' + str(total_sample) + '_' + str(total_class) + '_' + str(reduce_dim) + '.csv', data_stack, delimiter=',')
	print(np.shape(data_stack))