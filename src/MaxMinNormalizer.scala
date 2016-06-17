import org.apache.spark.Logging
import org.apache.spark.annotation.{Experimental, DeveloperApi}
import org.apache.spark.mllib.linalg.{Vectors, DenseVector, Vector}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD

/**
  * Created by yaozy on 2016/6/14.
  */

class MaxMinScaler(toMax: Double, toMin: Double) extends Logging{
  def this() = this(1.0,0.0)

  if (toMax < toMin){
    logWarning("toMax and/or toMin are not set correctly. maxValue should be larger than minValue")
  }

  /**
    * Computers the maximum and minimum and stores as a model to be used for later normalization
    *
    * @param data The data used to compute the maximum and minimum to build the transformation model
    * @return a MaxMinScalerModel
    */
  def fit(data: RDD[Vector]): MaxMinScalerModel = {
    val summary = data.treeAggregate(new MultivariateOnlineSummarizer)(
      (aggregator, data) => aggregator.add(data),
      (aggregator1, aggregator2) => aggregator1.merge(aggregator2))
    new MaxMinScalerModel(summary.max,summary.min,toMax,toMin)
  }
}

@Experimental
class MaxMinScalerModel(
    val fromMax: Vector,
    val fromMin: Vector,
    var toMax: Double,
    var toMin: Double) extends Serializable {

  def this(fromMax: Vector, fromMin: Vector) = {
    this(fromMax, fromMin, toMax = 1.0, toMin = 0.0)
    require(fromMax.size == fromMin.size,
      "fromMax and fromMin must have equal size if both are provided."
    )
  }

  def this(fromMax: Vector, fromMin: Vector, toMax: Double) = this(fromMax, fromMin, toMax, toMax - 1.0)

  @DeveloperApi
  def setToMax(toMax: Double): this.type = {
    require(toMax > this.toMin, "cannot set toMax less than toMin.")
    this.toMax = toMax
    this
  }

  @DeveloperApi
  def setToMin(toMin: Double): this.type = {
    require(toMin < this.toMax, "cannot set toMin larger than toMax.")
    this.toMin = toMin
    this
  }

  def transform(vector: Vector): Vector = {
    require(fromMax.size == vector.size)
    vector match {
      case DenseVector(vs) =>
        val values = vs.clone()
        val size = values.length
        var i = 0
        while (i < size) {
          values(i) = if (fromMax(i) != fromMin(i))
              (toMax - toMin) / (fromMax(i) - fromMin(i)) * (values(i) - fromMax(i)) + toMax
            else
              (toMax + toMin) / 2.0
          i += 1
        }
        Vectors.dense(values)
      case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
    }
  }

  def transform(data: RDD[Vector]): RDD[Vector] = {
    data.map(line => this.transform(line))
  }

}

