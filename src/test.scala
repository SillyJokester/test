import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy}

import scala.math.random

import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD, LogisticRegressionWithSGD}

/** Computes an approximation to pi */
object RemoteDebug {

  def combineTwoRDD(RDDLeft: RDD[String],RDDRight: RDD[String]) ={
    val pairRDDLeft = RDDLeft.map(x => (x.split(",")(0),x.split(x.split(",")(0) + ",")(1)))
    val pairRDDRight = RDDRight.map(x => (x.split(",")(0),x.split(x.split(",")(0) + ",")(1)))
    pairRDDLeft.join(pairRDDRight)
  }

  def getScalingRDD(vec: RDD[Vector]): RDD[Vector] = {
    new StandardScaler(withMean = true, withStd = true).fit(vec).transform(vec)
    //scaler.transform(vec)
  }

  def main1(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
    val spark = new SparkContext(conf)
    val slices = if (args.length > 0) args(0).toInt else 2
    val n = math.min(100000L * slices, Int.MaxValue).toInt // avoid overflow
    val count = spark.parallelize(1 until n, slices).map { i =>
        val x = random * 2 - 1
        val y = random * 2 - 1
        if (x*x + y*y < 1) 1 else 0
      }.reduce(_ + _)
    println("Pi is roughly " + 4.0 * count / n)
    spark.stop()
  }

  def main(args: Array[String]): Unit ={
    val conf = new SparkConf().setAppName("MLtest").setMaster("local")
    val sc = new SparkContext(conf)
    val entstate = sc.textFile("E:/my projects/IdeaProject/test/src/data/t_entstate_1.csv")
    val entinfo = sc.textFile("E:/my projects/IdeaProject/test/src/data/entinfo.csv")
    //val entstaterdd = entstate.map(line => (line.split(",")(0),line.split(",")(1)))
    //val entinfordd = entinfo.map(line => (line.split(",")(0),line.split(line.split(",")(0) + ",")(1)))
    //entstate.take(10).foreach(println)
    //entinfordd.take(10).foreach(println)
    val combinerdd = combineTwoRDD(entstate,entinfo)
    //combinerdd.take(10).foreach(println)
    val dataset = combinerdd.map(
      line =>
        LabeledPoint(
          line._2._1.toDouble,
          Vectors.dense(
            line._2._2.split(",")(0).toDouble,
            line._2._2.split(",")(1).toDouble,
            line._2._2.split(",")(2).toDouble,
            line._2._2.split(",")(3).toDouble
          )))
    //dataset.take(10).foreach(println)
    val splits = dataset.randomSplit(Array(0.6,0.4), seed = 11L)
    val tsplit = splits(0)
    val vsplit = splits(1)
    //Logistic Regression
    //val model = LogisticRegressionWithSGD.train(tsplit,numIterations = 50)

    // Random Forest
    /*
    val model = RandomForest.trainClassifier(
      input = tsplit,
      numClasses = 2,
      categoricalFeaturesInfo = Map[Int,Int](),
      numTrees = 3,
      featureSubsetStrategy = "auto",
      impurity = "entropy",
      maxDepth = 5,
      maxBins = 100
    )
    */

    // SVM with SGD
    //val model = SVMWithSGD.train(tsplit, numIterations = 50)

    // Navie Bayes
    //val model = NaiveBayes.train(tsplit)

    // Decision Tree
    val model = DecisionTree.train(tsplit, Algo.Classification, Entropy, 5)

    // GBDT
    /*
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(50)
    boostingStrategy.getTreeStrategy().setNumClasses(2)
    boostingStrategy.getTreeStrategy().setMaxDepth(5)
    val model = GradientBoostedTrees.train(tsplit,boostingStrategy)
    */

    val predictionAndLabels = vsplit.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)
  }

  def main3(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sc = new SparkContext(conf)
    val testdataset = sc.parallelize(List((1,4),(2,3),(3,2),(4,1)))
    println(testdataset.count())
    val vecdataset = testdataset.map(line =>
      Vectors.dense(line._1.toDouble,line._2.toDouble)
    )
    getScalingRDD(vecdataset).foreach(println)
    val pairtestdataset = testdataset.map(line => (line._1,Vectors.dense(line._2.toDouble)))
    val scaleddataset = getScalingRDD(pairtestdataset.map(line => line._2))
    val flagdataset = pairtestdataset.map(line => Vectors.dense(line._1))
    scaleddataset.foreach(println)
    flagdataset.union(scaleddataset).foreach(println)
  }
}

