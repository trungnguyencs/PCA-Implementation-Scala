

var mu1:Array[Int] = new Array[Int](3)
var mu1 = Array(0, 0, 0)

var mu2:Array[Int] = new Array[Int](3)
var mu2 = Array(1, 1, 1)

var rows = 3
var cols = 3
var cov_mat1 = Array.ofDim[Int](rows,cols)
cov_mat1 = Array(Array(1,0,0), Array(0,1,0), Array(0,0,1))

for {
     | i <- 0 until rows
     | j <- 0 until cols
     | }
     | println(s"($i)($j) = ${cov_mat1(i)(j)}")


var cov_mat2 = Array.ofDim[Int](rows,cols)
cov_mat2 = Array(Array(1,0,0), Array(0,1,0), Array(0,0,1))

for {
     | i <- 0 until rows
     | j <- 0 until cols
     | }
     | println(s"($i)($j) = ${cov_mat2(i)(j)}")


import org.apache.commons.math3.distribution

//class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
class1_sample = MultivariateNormalDistribution(mu1, double[][] cov_mat1)

