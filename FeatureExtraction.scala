package com.spark.recommendation

import org.apache.spark.{sql, SparkConf}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Dataset, SparkSession}


object FeatureExtraction {

  val spark = SparkSession.builder.master("local[2]").appName("FeatureExtraction").getOrCreate()

  case class Popularity(userId: Int, ItemId: Int, score: Float, timestamp: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("\t")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }
  def getFeatures(): sql.DataFrame = {
    import spark.implicits._
    val ratings = spark.read.textFile("/Users/yosra/test.data").map(parseRating).toDF()
    println(ratings.first())
    return popularuty
  }

  def getSpark(): SparkSession = {
    return spark
  }

  def main(args: Array[String]) {
    getFeatures()
  }

}