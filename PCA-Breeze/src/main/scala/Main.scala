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
  
  // Geting the mean vector
  val mean_vec = mean(data(::, *)) 

  // Computing the Scatter Matrix
  var scatter_mat = DenseMatrix.zeros[Double](dim, dim)	
  for (i <- 0 until total_sample) {
    scatter_mat += (data(i,::) - mean_vec).t*(data(i,::) - mean_vec)
  }
  // println(scatter_mat)

  val es = eigSym(scatter_mat)
  val eig_val = es.eigenvalues
  val eig_mat = es.eigenvectors
  println(eig_val)
  println(eig_mat)
}

