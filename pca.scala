import org.apache.commons.math3.distribution.MultivariateNormalDistribution

object Main extends App {

  var mu1:Array[Double] = new Array[Double](3)
  mu1 = Array(0, 0, 0)

  var mu2:Array[Double] = new Array[Double](3)
  mu2 = Array(1, 1, 1)

  var rows = 3
  var cols = 3

  var cov_mat1 = Array.ofDim[Double](rows,cols)
  cov_mat1 = Array(Array(1,0,0), Array(0,1,0), Array(0,0,1))

  for {
     i <- 0 until rows
     j <- 0 until cols
  }
    println(s"($i)($j) = ${cov_mat1(i)(j)}")


  var cov_mat2 = Array.ofDim[Double](rows,cols)
  cov_mat2 = Array(Array(1,0,0), Array(0,1,0), Array(0,0,1))

  for {
    i <- 0 until rows
    j <- 0 until cols
  }
    println(s"($i)($j) = ${cov_mat2(i)(j)}")

  var class1_sample = new MultivariateNormalDistribution(mu1, cov_mat1)


}
