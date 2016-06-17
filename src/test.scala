import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.model.Predict
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy}


import scala.math.random

import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD, LogisticRegressionWithSGD, LogisticRegressionWithLBFGS}


/** Computes an approximation to pi */
object test {

  def combineTwoRDD(RDDLeft: RDD[String],RDDRight: RDD[String]) ={
    val pairRDDLeft = RDDLeft.map(x => (x.split(",")(0),x.split(x.split(",")(0) + ",")(1)))
    val pairRDDRight = RDDRight.map(x => (x.split(",")(0),x.split(x.split(",")(0) + ",")(1)))
    pairRDDLeft.join(pairRDDRight)
  }

  def getScalingRDD(vec: RDD[Vector]): RDD[Vector] = {
    new StandardScaler(withMean = true, withStd = true).fit(vec).transform(vec)
    //scaler.transform(vec)
  }
  def getMaxMinRDD(vec: RDD[Vector]): RDD[Vector] = {
    new MaxMinScaler(toMax = 1.0,toMin = 0.2).fit(vec).transform(vec)
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
    val entstate = sc.textFile("src/data/t_entstate_1.csv")
    val entinfo = sc.textFile("src/data/entinfo.csv")
    //val entstaterdd = entstate.map(line => (line.split(",")(0),line.split(",")(1)))
    //val entinfordd = entinfo.map(line => (line.split(",")(0),line.split(line.split(",")(0) + ",")(1)))
    //entstate.take(10).foreach(println)
    //entinfordd.take(10).foreach(println)
    val combinerdd = combineTwoRDD(entstate,entinfo)
    //combinerdd.take(10).foreach(println)
    /*
    val dataset = combinerdd.map(
      line =>
        LabeledPoint(
          line._2._1.toDouble,
          Vectors.dense(
            line._2._2.split(",").map(_.toDouble)
          )))
    //dataset.take(10).foreach(println)
    */

    val labelset = combinerdd.map(
      line =>
        line._2._1.toInt
      )
    val featureset = combinerdd.map(
      line =>
        Vectors.dense(
          line._2._2.split(",").map(_.toDouble)
        )
    )

    /*
    val dataset = labelset.zip(getMaxMinRDD(featureset)).map(
      line =>
        LabeledPoint(line._1,line._2)
    )
    */

    val dataset = labelset.zip(getScalingRDD(featureset)).map(
      line =>
        LabeledPoint(line._1,line._2)
    )

    val splits = dataset.randomSplit(Array(0.6,0.4), seed = 12L)
    val tsplit = splits(0).cache()
    val vsplit = splits(1)

    //Logistic Regression
    val model = LogisticRegressionWithSGD.train(tsplit,numIterations = 50)

    /*
    val model = new LogisticRegressionWithLBFGS().
      setNumClasses(2).
      setIntercept(true).
      run(tsplit)
    */
    /*
    val lrmodel = new LogisticRegressionWithSGD()
    lrmodel.optimizer.
      setNumIterations(50).
      setRegParam(0.3)
    val model = lrmodel.run(tsplit)
    */

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
    //val model = DecisionTree.train(tsplit, Algo.Classification, Entropy, 5)

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
    predictionAndLabels.take(10).foreach(println)
    /*
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Precision = " + precision)
    */

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metrics.areaUnderROC()
    val auPR = metrics.areaUnderPR()
    println("Area under ROC: " + auROC + "; Area under PR: " + auPR)

    val vecweights = model.weights
    val intercept = model.intercept
    val probAndLabel = vsplit.map{
      case LabeledPoint(label, features) =>
        val margin = vecweights(0) * features(0) +
          vecweights(1) * features(1) +
          vecweights(2) * features(2) +
          vecweights(3) * features(3) + intercept
        val prob = 1.0 / (1.0 +  math.exp(-margin))
        val prediction = model.predict(features)
        (prob, prediction, label,features)
    }
    probAndLabel.take(100).foreach(println)
    println(model.weights)


    //println("Stay rate in train set: " + (tsplit.filter(line => line.label > 0).count().toDouble / tsplit.count().toDouble))
    //println("Stay rate in test set: " + (vsplit.filter(line => line.label > 0).count().toDouble / vsplit.count().toDouble))
  }

  def main3(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sc = new SparkContext(conf)
    val testdataset = sc.parallelize(List((1,4,1),(2,3,2),(3,2,3),(4,1,4)))
    val tmpdata = testdataset.map(line => (line.productElement(1),line.productElement(0),line.productElement(2)))

    val pairtestdataset = testdataset.map(line => (line._1,Vectors.dense(line._2.toDouble,line._3.toDouble)))
    val scaleddataset = getScalingRDD(pairtestdataset.map(line => line._2))
    val flagdataset = pairtestdataset.map(line => line._1)
    scaleddataset.foreach(println)
    val tmpset = pairtestdataset.map(line => line._2)
    tmpset.foreach(println)
    val testmmn = new MaxMinScaler().fit(tmpset).transform(tmpset)
    testmmn.foreach(println)
    /*
    val accv = sc.accumulator(0)
    val flagdata = flagdataset.map {
      line =>
        accv += 1
        (accv, line)
    }.map(line => (line._1.toString(),line._2))
    flagdata.foreach(println)

    accv.setValue(0)
    val scaleddata = scaleddataset.map{
      line =>
        accv += 1
        (accv,line(0),line(1))
    }.map(line => (line._1.toString(),Vectors.dense(line._2,line._3)))
    scaleddata.foreach(println)

    flagdata.join(scaleddata).foreach(println)
    val findata = flagdata.join(scaleddata).map(line =>
      (line._2._1,line._2._2))
    findata.foreach(println)
    */
  }

  def main4(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("MLtest").setMaster("local")
    val sc = new SparkContext(conf)
    val entstate = sc.textFile("E:/my projects/IdeaProject/test/src/data/t_entstate_1.csv")
    val entinfo = sc.textFile("E:/my projects/IdeaProject/test/src/data/entinfo.csv")
    //val entstaterdd = entstate.map(line => (line.split(",")(0),line.split(",")(1)))
    //val entinfordd = entinfo.map(line => (line.split(",")(0),line.split(line.split(",")(0) + ",")(1)))
    //entstate.take(10).foreach(println)
    //entinfordd.take(10).foreach(println)
    val combinerdd = combineTwoRDD(entstate, entinfo)
    val labelset = combinerdd.map(
      line =>
        line._2._1
    )
    val featureset = combinerdd.map(
      line =>
        Vectors.dense(
          line._2._2.split(",").map(_.toDouble)
        )
    )

    val scaledfeaset = getScalingRDD(featureset)

    val accv = sc.accumulator(0)
    val tmplabelset = labelset.map {
      line =>
        accv += 1
        (accv.toString(), line)
    }
    tmplabelset.take(10).foreach(println)
    accv.setValue(0)
    val tmpfeatureset = scaledfeaset.map{
      line =>
        accv += 1
        (accv.toString(),line)
    }
    tmpfeatureset.take(10).foreach(println)
    val dataset = tmplabelset.join(tmpfeatureset).map(line =>
      LabeledPoint(line._2._1.toDouble,line._2._2))

    val splits = dataset.randomSplit(Array(0.6,0.4), seed = 11L)
    val tsplit = splits(0).cache()
    val vsplit = splits(1)

    //Logistic Regression
    val model = LogisticRegressionWithSGD.train(tsplit,numIterations = 50)
    val predictionAndLabels = vsplit.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metrics.areaUnderROC()
    val auPR = metrics.areaUnderPR()
    metrics.roc()
    println("Area under ROC: " + auROC + "; Area under PR: " + auPR)
    val vecweights = model.weights
    val intercept = model.intercept
    val probAndLabel = vsplit.map{
      case LabeledPoint(label, features) =>
        val prob = vecweights(0) * features(0) +
          vecweights(1) * features(1) +
          vecweights(2) * features(2) +
          vecweights(3) * features(3) + intercept
        val prediction = model.predict(features)
        (prob, prediction, label,features)
    }
    probAndLabel.take(100).foreach(println)
    println(model.weights)
  }

}

