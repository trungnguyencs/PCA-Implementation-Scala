# PCA-Implementation-Scala

An implementation of PCA algorithm from scratch in Scala Breeze and Python numpy. 

### Summarizing:

* Compute the d-dimensional mean vector (i.e., the means for every dimension of the whole dataset)
* Compute the scatter matrix (alternatively, the covariance matrix) of the whole data set
* Compute eigenvectors (e_1, e_2,..., e_d) and corresponding eigenvalues (λ_1, λ_2,..., λ_d)
* Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d\*k dimensional matrix W (where every column represents an eigenvector)
* Use this d\*k eigenvector matrix to transform the samples onto the new subspace

### Install sbt:
`echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list`
`curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add`
`sudo apt-get update`
`sudo apt-get install sbt`

### Run the program:
`cd` to `PCA-Scala`
`sbt run`

 

### Screenshot:
![alt text](https://github.com/trungnguyencs/PCA-Implementation-Scala/blob/master/git_img/run.png "Screenshot")


