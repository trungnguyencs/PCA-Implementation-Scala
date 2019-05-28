import breeze.linalg._, eigSym.EigSym
import breeze.math._
import breeze.numerics._
import breeze.stats.distributions.MultivariateGaussian
import breeze.stats.mean

object Main extends App {

  // Generate data input
  val dim = 4
  val total_class = 2
  val total_sample = 16  
  val reduce_dim = 2

  var data = DenseMatrix.zeros[Double](total_sample*total_class, dim)	
  val cov_mat = DenseMatrix.eye[Double](dim)

  for (i <- 0 until total_class) {
    var mean_vec = DenseVector.ones[Double](dim).map(value => value * i)
    var distr = new MultivariateGaussian(mean_vec, cov_mat)
    for (j <- 0 until total_sample) {
      data(j+i*total_sample, ::) := distr.sample().t
    }
  }  

  println(s"\nNumber of classes: ${total_class}")
  println(s"Number of samples in each class: ${total_sample}")
  println(s"Number of dimmensions: ${dim}")
  println(s"Number of reduced dimmensions: ${reduce_dim}")
  println(s"All data shape: ${data.rows} rows, ${data.cols} columns")
  
  // Geting the mean vector
  val mean_vec = mean(data(::, *)) 

  println(s"\nMean vector:\n${mean_vec}")

  // Computing the Scatter Matrix
  var scatter_mat = DenseMatrix.zeros[Double](dim, dim)	
  for (i <- 0 until total_sample) {
    scatter_mat += (data(i,::) - mean_vec).t*(data(i,::) - mean_vec)
  }
  println(s"\nScatter matrix:\n${scatter_mat}")

  // Computing the Eigenvalues and Eigenvectors
  val es = eigSym(scatter_mat)
  val eig_val = es.eigenvalues
  val eig_mat = es.eigenvectors
  println(s"\nEigen values:\n${eig_val}")
  println(s"\nEigen matrix:\n${eig_mat}")

  // Sort the Eigenvalues in decreasing order
  val keep_idx = argtopk(abs(eig_val), dim-reduce_dim)
  println(s"\nKeep indexes:\n${keep_idx}")

  // Reduce (reduce_dim) dimensions: Take only (dim-reduce_dim) scatter vectors to put in the new eigen matrix
  var new_eig_mat = DenseMatrix.zeros[Double](dim, dim-reduce_dim)	
  for (i <- 0 until keep_idx.length) {
    new_eig_mat(::, i) := eig_mat(::, keep_idx(i))
  }
  println(s"\nNew eigen matrix:\n${new_eig_mat}")

}


