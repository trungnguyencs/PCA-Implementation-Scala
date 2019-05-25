import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.MultivariateGaussian

object Main extends App {
  val dim = 4
  val total_class = 2
  val total_sample = 100  
  val reduce_dim = 2

  val cov_mat = DenseMatrix.eye[Double](dim)
  var mean_vec = DenseVector.ones[Double](dim)
  var mulVal = new MultivariateGaussian(mean_vec, cov_mat)
  val class_sample = mulVal.draw()
  print(class_sample)
}

