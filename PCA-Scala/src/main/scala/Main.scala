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

  /** Get csv data file name list */
  def getListOfFiles(dir: String): List[String] = {
    val file = new File(dir)
    file.listFiles.filter(_.isFile)
      .filter(_.getName.endsWith("csv"))
      .map(_.getPath).toList
  }

  /** Get list of part containing dim, total_sample, total_class, reduced_dim from csv file name  */
  def trimFileList(f_list: List[String]) : List[String] = {
    var trimmed_f_list = f_list.map(_.split("[/.]").map(_.trim)).map(f_list => f_list(4))
    return trimmed_f_list
  }

  /** Return the dataset in DenseMatrix object from reading csv file */
  def readCsv(f_name: String) : DenseMatrix[Double] = {
    val data = csvread(new File(path + f_name + ".csv"),',').t
    return data
  }  

  /** Generate data using a Multivariate Distribution */
    // var data = DenseMatrix.zeros[Double](total_sample*total_class, dim)	
    // val cov_mat = DenseMatrix.eye[Double](dim)

    // for (i <- 0 until total_class) {
    //   var mean_vec = DenseVector.ones[Double](dim).map(value => value * i)
    //   var distr = new MultivariateGaussian(mean_vec, cov_mat)
    //   for (j <- 0 until total_sample) {
    //     data(j+i*total_sample, ::) := distr.sample().t
    //   }
    // }  

  /** Get the dim, total_sample, total_class, reduced_dim information from csv file name */
  def extractFileName(f_name: String) : (Int, Int, Int, Int) = {
    val inp_size = f_name.split("_").map(_.trim)
    val dim = inp_size(0).toInt
    val total_sample = inp_size(1).toInt
    val total_class = inp_size(2).toInt
    val reduced_dim = inp_size(3).toInt
    return (dim, total_sample, total_class, reduced_dim)
  }

  /** Geting the mean vector from the whole dataset */
  def computeMeanVector(data: DenseMatrix[Double]) : Transpose[DenseVector[Double]] = {
    val mean_vec = mean(data(::, *))     
    return mean_vec
  }

  /** Computing the Scatter Matrix from the whole dataset */
  def computeScatterMatrix(data:DenseMatrix[Double], mean_vec:Transpose[DenseVector[Double]],
   dim:Int, total_sample:Int) : DenseMatrix[Double] = {
    var scatter_mat = DenseMatrix.zeros[Double](dim, dim)	
    for (i <- 0 until total_sample) {
      scatter_mat += (data(i,::) - mean_vec).t*(data(i,::) - mean_vec)
    }
    return scatter_mat
  }

  /** Computing the eigenvalues and eigenvectors from the scatter matrix */
  def computeEig(scatter_mat:DenseMatrix[Double]) : (DenseVector[Double], DenseMatrix[Double]) = {
    val es = eigSym(scatter_mat)
    val eig_val = es.eigenvalues
    val eig_mat = es.eigenvectors
    return (eig_val, eig_mat)
  }

  /** Reduce (reduced_dim) dimensions: Take only (dim-reduced_dim) scatter vectors to put in the new eigen matrix */
  def reduceDim(eig_mat:DenseMatrix[Double], keep_idx:IndexedSeq[Int], dim:Int, reduced_dim:Int) : DenseMatrix[Double] = {
    var new_eig_mat = DenseMatrix.zeros[Double](dim, dim-reduced_dim)	
    for (i <- 0 until keep_idx.length) {
      new_eig_mat(::, i) := eig_mat(::, keep_idx(i))
    }
    return new_eig_mat
  }

  /** Project the original dataset to the new subspace using the remaining eigenvectors */
  def projectData(data:DenseMatrix[Double], new_eig_mat:DenseMatrix[Double]) : DenseMatrix[Double] = {
    val proj_data = data*new_eig_mat
    return proj_data
  }

  val path = "../data/"

  // Trim name
  val f_list = getListOfFiles(path)
  var trimmed_f_list = trimFileList(f_list)

  // Write file 
  val pw = new PrintWriter(new File("../output_scala.csv" ))
  pw.write(s"dim,total_sample,total_class,reduced_dim,data_time,total_time,memory_usage}\n")

  for(f_name <- trimmed_f_list) {
    println(s"\n- File name: ${f_name}")

    // Get the initial time stamp 
    val t1 = System.nanoTime

    // Extract data input
    val (dim, total_sample, total_class, reduced_dim) = extractFileName(f_name)
    println(s"\n- Number of dimmensions: ${dim}")
    println(s"- Number of samples in each class: ${total_sample}")
    println(s"- Number of classes: ${total_class}")
    println(s"- Number of reduced dimmensions: ${reduced_dim}")

    // Get the dataset from reading csv file
    val data = readCsv(f_name)
    println(s"- All data shape: ${data.rows} rows, ${data.cols} columns\n")  

    // Get the timestamp right after finishing reading the csv file
    val data_time = (System.nanoTime - t1) / 1e6d

    // Geting the mean vector
    val mean_vec = computeMeanVector(data) 
    println(s"- Mean vector:\n${mean_vec}\n")

    // Computing the Scatter Matrix
    val scatter_mat = computeScatterMatrix(data, mean_vec, dim, total_sample)
    println(s"- Scatter matrix:\n${scatter_mat}\n")

    // Computing the Eigenvalues and Eigenvectors
    val (eig_val, eig_mat) = computeEig(scatter_mat) 
    println(s"- Eigenvalues:\n${eig_val}\n")
    println(s"- Eigenvectors:\n${eig_mat}\n")

    // Sort the Eigenvalues in decreasing order
    val keep_idx = argtopk(abs(eig_val), dim-reduced_dim)
    println(s"- Keep indexes:\n${keep_idx}\n")
    println(keep_idx.getClass)

    // Reduce (reduced_dim) dimensions: Take only (dim-reduced_dim) scatter vectors to put in the new eigen matrix    
    val new_eig_mat = reduceDim(eig_mat, keep_idx, dim, reduced_dim)
    println(s"- New eigenvectors:\n${new_eig_mat}\n")

    // Project all samples to the new subspace:
    val proj_data = projectData(data, new_eig_mat)
    println(s"- Projected data shape: ${proj_data.rows} rows, ${proj_data.cols} columns")

    // Calculate load data runtime and total runtime 
    val total_time = (System.nanoTime - t1) / 1e6d
    println(s"\n--- Load data runtime: ${data_time} miliseconds --- ")
    println(s"--- Total runtime: ${total_time} miliseconds --- ")

    // Calculate memory usage
    val mb = 1024 * 1024
    val runtime = Runtime.getRuntime
    val memory_usage = (runtime.totalMemory - runtime.freeMemory) / mb
    println(s"--- Memory usage: ${memory_usage} Mbytes --- \n")

    // Write each line to output file 
    pw.write(s"${dim},${total_sample},${total_class},${reduced_dim},${data_time},${total_time},${memory_usage}\n")
  }   
  pw.close
}