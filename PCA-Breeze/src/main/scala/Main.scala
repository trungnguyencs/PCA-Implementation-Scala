import breeze.linalg._, eigSym.EigSym
import breeze.math._
import breeze.numerics._
import breeze.stats.distributions.MultivariateGaussian
import breeze.stats.mean
import java.io._
import java.io.File
import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter

object Main extends App {

  val path = "../data/"

  /** Provides a service as described.
   *
   *  This is further documentation of what we're documenting.
   *  Here are more details about how it works and what it does.
   */

  // Read file
  def getListOfFiles(dir: String): List[String] = {
    val file = new File(dir)
    file.listFiles.filter(_.isFile)
      .filter(_.getName.endsWith("csv"))
      .map(_.getPath).toList
  }




  // Trim name
  val f_list = getListOfFiles(path)
  var trimmed_f_list = f_list.map(_.split("[/.]").map(_.trim)).map(f_list => f_list(4))

  // Write file 
  val pw = new PrintWriter(new File("../output_scala.csv" ))
  pw.write(s"dim,total_sample,total_class,reduce_dim,data_time,total_time,memory_usage}\n")


  for(f_name <- trimmed_f_list) {
    val t1 = System.nanoTime

    println(s"\n- File name: ${f_name}")

    val inp_size = f_name.split("_").map(_.trim)
    val data = csvread(new File(path + f_name + ".csv"),',').t

    // Extract data input
    val dim = inp_size(0).toInt
    val total_sample = inp_size(1).toInt
    val total_class = inp_size(2).toInt
    val reduce_dim = inp_size(3).toInt
    
    val data_time = (System.nanoTime - t1) / 1e6d

    // var data = DenseMatrix.zeros[Double](total_sample*total_class, dim)	
    // val cov_mat = DenseMatrix.eye[Double](dim)

    // for (i <- 0 until total_class) {
    //   var mean_vec = DenseVector.ones[Double](dim).map(value => value * i)
    //   var distr = new MultivariateGaussian(mean_vec, cov_mat)
    //   for (j <- 0 until total_sample) {
    //     data(j+i*total_sample, ::) := distr.sample().t
    //   }
    // }  

    println(s"\n- Number of dimmensions: ${dim}")
    println(s"- Number of samples in each class: ${total_sample}")
    println(s"- Number of classes: ${total_class}")
    println(s"- Number of reduced dimmensions: ${reduce_dim}")
    println(s"- All data shape: ${data.rows} rows, ${data.cols} columns\n")
    
    // Geting the mean vector
    val mean_vec = mean(data(::, *)) 

    println(s"- Mean vector:\n${mean_vec}\n")

    // Computing the Scatter Matrix
    var scatter_mat = DenseMatrix.zeros[Double](dim, dim)	
    for (i <- 0 until total_sample) {
      scatter_mat += (data(i,::) - mean_vec).t*(data(i,::) - mean_vec)
    }
    println(s"- Scatter matrix:\n${scatter_mat}\n")

    // Computing the Eigenvalues and Eigenvectors
    val es = eigSym(scatter_mat)
    val eig_val = es.eigenvalues
    val eig_mat = es.eigenvectors
    println(s"- Eigenvalues:\n${eig_val}\n")
    println(s"- Eigenvectors:\n${eig_mat}\n")

    // Sort the Eigenvalues in decreasing order
    val keep_idx = argtopk(abs(eig_val), dim-reduce_dim)
    println(s"- Keep indexes:\n${keep_idx}\n")

    // Reduce (reduce_dim) dimensions: Take only (dim-reduce_dim) scatter vectors to put in the new eigen matrix
    var new_eig_mat = DenseMatrix.zeros[Double](dim, dim-reduce_dim)	
    for (i <- 0 until keep_idx.length) {
      new_eig_mat(::, i) := eig_mat(::, keep_idx(i))
    }
    println(s"- New eigenvectors:\n${new_eig_mat}\n")

    // Project all samples to the new subspace:
    val proj_data = data*new_eig_mat
    println(s"- Projected data shape: ${proj_data.rows} rows, ${proj_data.cols} columns")

    val total_time = (System.nanoTime - t1) / 1e6d

    println(s"\n--- Load data runtime: ${data_time} miliseconds --- ")
    println(s"--- Total runtime: ${total_time} miliseconds --- ")

    val mb = 1024 * 1024
    val runtime = Runtime.getRuntime
    val memory_usage = (runtime.totalMemory - runtime.freeMemory) / mb
    println(s"--- Memory usage: ${memory_usage} Mbytes --- \n")

    // Write each line to output file 
    pw.write(s"${dim},${total_sample},${total_class},${reduce_dim},${data_time},${total_time},${memory_usage}\n")
  }   
  pw.close
}


