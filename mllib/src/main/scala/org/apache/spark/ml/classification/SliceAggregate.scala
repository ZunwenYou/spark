// scalastyle:off

package org.apache.spark.ml.classification

import java.util.Random

import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.ml.rdd.RDDFunctions._
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD

/**
 * Created by kevinzwyou on 17-1-22.
 */
object SliceAggregator {

  def parseArgs(args: Array[String]): Map[String, String] = {
    val argsMap = new scala.collection.mutable.HashMap[String, String]
    args.foreach(s => {
      val str = s.trim
      val pos = str.indexOf(":")
      if (pos != -1) {
        val k = str.substring(0, pos)
        val v = str.substring(pos + 1)
        argsMap.put(k, v)
        println(s"args: $k -> $v")
      }
    })
    argsMap.toMap
  }

  def generateData(sampleNum: Int, featLength: Int, partitionNum: Int): RDD[Instance] = {

    def randVector(rand: Random, dim: Int): Vector = {
      Vectors.dense(Array.tabulate(dim)(i => rand.nextGaussian()))
    }

    val spark = SparkSession.builder().getOrCreate()

    val rand = new Random(42)
    val initModel = randVector(rand, featLength)
    println(s"init model: ${initModel.toArray.slice(0, 100).mkString(" ")}")

    val bcModel = spark.sparkContext.broadcast(initModel)

    spark.sparkContext.parallelize(0 until sampleNum, partitionNum)
      .map { id =>
        val rand = new Random(id)
        val feat = randVector(rand, featLength)

        val margin = BLAS.dot(feat, bcModel.value)
        val prob = 1.0 / (1.0 + math.exp(-1 * margin))

        val label = if (rand.nextDouble() > prob) 0.0 else 1.0
        Instance(label, 1.0, feat)
      }
  }

  def compare(instances: RDD[Instance], featNum: Int, slice: Int, depth: Int): (Long, Long) = {
    val thisInstances = if (instances.first().features.size == featNum) {
      instances
    } else {
      val thisInstances = instances.map { case Instance(label, weight, features) =>
        Instance(label, weight, Vectors.dense(features.toArray.slice(0, featNum)))
      }
      thisInstances.cache()
      thisInstances.count()
      thisInstances
    }

    // slice aggregate
    var startTime = System.currentTimeMillis()
    val summarizer = {
      val seqOp = (c: MultivariateOnlineSummarizer, instance: Instance) =>
        c.add(instance.features, instance.weight)
      val combOp = (c1: MultivariateOnlineSummarizer, c2: MultivariateOnlineSummarizer) =>
        c1.merge(c2)
      thisInstances.sliceAggregate(new MultivariateOnlineSummarizer)(seqOp, combOp, slice)
    }
    var endTime = System.currentTimeMillis()
    val sliceTime = endTime - startTime
    println(s"slice aggregate time: $sliceTime ms")

    // tree aggregate
    var treeTime = 0L
    try {
      startTime = System.currentTimeMillis()
      val summarizer1 = {
        val seqOp = (c: MultivariateOnlineSummarizer, instance: Instance) =>
          c.add(instance.features, instance.weight)
        val combOp = (c1: MultivariateOnlineSummarizer, c2: MultivariateOnlineSummarizer) =>
          c1.merge(c2)
        thisInstances.treeAggregate(new MultivariateOnlineSummarizer)(seqOp, combOp, depth)
      }
      endTime = System.currentTimeMillis()
      treeTime = endTime - startTime
      println(s"tree aggregate time: $treeTime ms")

    } catch {
      case e: Exception => println("treeAggregate fail. feat num: " + featNum)
    }

    thisInstances.unpersist()
    (sliceTime, treeTime)
  }


  def main(args: Array[String]) {
    val argsMap = parseArgs(args)
    val mode = argsMap.getOrElse("mode", "yarn-cluster")
    val featNum = argsMap.getOrElse("featNum", "100").toInt

    val sampleNum = 1000
    val partitionNum = 300

    val spark = SparkSession.builder()
      .master(mode)
      .appName(this.getClass.getSimpleName)
      .getOrCreate()
    Thread.sleep(20 * 1000)

    val instances = generateData(sampleNum, featNum, partitionNum)
    instances.cache()
    println(s"instance count: ${instances.count()} feat length: ${instances.first().features.size}")


    val n = 10
    val tenThousand = 10000
    val featLenArray = Array(1 * tenThousand, 10 * tenThousand, 100 * tenThousand, 250 * tenThousand, 500 * tenThousand, 750 * tenThousand, 1000 * tenThousand, 2000 * tenThousand)

    featLenArray.foreach { thisFeatNum =>
      val thisSlice = if (thisFeatNum > tenThousand) math.ceil(thisFeatNum / tenThousand).toInt else 1
      val thisDepth = if (thisFeatNum < 500 * tenThousand) 2 else if (thisFeatNum < 1000 * tenThousand) 3
      else if (thisFeatNum < 2000 * tenThousand) 4 else 5

      val timeCosts = (0 until n).map { i =>
        println(s"counter: $i")
        compare(instances, thisFeatNum, thisSlice, thisDepth)
      }
      println(s"featNum: $thisFeatNum")
      println(s"slice time: ${timeCosts.map(_._1).sum / n}")
      println(s"tree time: ${timeCosts.map(_._2).sum / n}")
    }
  }
}
// scalastyle:on