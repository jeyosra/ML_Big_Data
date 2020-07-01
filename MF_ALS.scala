package org.apache.spark.mllib.recommendation

import org.apache.spark.Logging
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.recommendation.{ALS2 => NewALS2}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class ALS  (
    private var numUserBlocks: Int,
    private var numProductBlocks: Int,
    private var rank: Int,
    private var iterations: Int,
    private var lambda: Double,
    private var implicitPrefs: Boolean,
    private var alpha: Double,
    private var seed: Long = System.nanoTime()
  ) extends Serializable with Logging {

  
  def this() = this(-1, -1, 10, 10, 0.01, false, 1.0)

  private var nonnegative = false

  private var intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK
  private var finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  private var checkpointInterval: Int = 10

 

  def setBlocks(numBlocks: Int): this.type = {
    this.numUserBlocks = numBlocks
    this.numProductBlocks = numBlocks
    this
  }


  def setUserBlocks(numUserBlocks: Int): this.type = {
    this.numUserBlocks = numUserBlocks
    this
  }

  def setProductBlocks(numProductBlocks: Int): this.type = {
    this.numProductBlocks = numProductBlocks
    this
  }


  def setRank(rank: Int): this.type = {
    this.rank = rank
    this
  }


  def setIterations(iterations: Int): this.type = {
    this.iterations = iterations
    this
  }


  def setLambda(lambda: Double): this.type = {
    this.lambda = lambda
    this
  }

  def setImplicitPrefs(implicitPrefs: Boolean): this.type = {
    this.implicitPrefs = implicitPrefs
    this
  }


  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }


  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

 
  def setNonnegative(b: Boolean): this.type = {
    this.nonnegative = b
    this
  }

  def setIntermediateRDDStorageLevel(storageLevel: StorageLevel): this.type = {
    require(storageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    this.intermediateRDDStorageLevel = storageLevel
    this
  }


  def setFinalRDDStorageLevel(storageLevel: StorageLevel): this.type = {
    this.finalRDDStorageLevel = storageLevel
    this
  }


  def setCheckpointInterval(checkpointInterval: Int): this.type = {
    this.checkpointInterval = checkpointInterval
    this
  }


  def run(ratings: RDD[Rating]): MatrixFactorizationModel = {
    val sc = ratings.context

    val numUserBlocks = if (this.numUserBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.size / 2)
    } else {
      this.numUserBlocks
    }
    val numProductBlocks = if (this.numProductBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.size / 2)
    } else {
      this.numProductBlocks
    }

    val (floatUserFactors, floatProdFactors) = NewALS2.train[Int](
      ratings = ratings.map(r => NewALS2.Rating(r.user, r.product, r.rating.toFloat)),
      rank = rank,
      numUserBlocks = numUserBlocks,
      numItemBlocks = numProductBlocks,
      maxIter = iterations,
      regParam = lambda,
      implicitPrefs = implicitPrefs,
      alpha = alpha,
      nonnegative = nonnegative,
      intermediateRDDStorageLevel = intermediateRDDStorageLevel,
      finalRDDStorageLevel = StorageLevel.NONE,
      checkpointInterval = checkpointInterval,
      seed = seed)

    val userFactors = floatUserFactors
      .mapValues(_.map(_.toDouble))
      .setName("users")
      .persist(finalRDDStorageLevel)
    val prodFactors = floatProdFactors
      .mapValues(_.map(_.toDouble))
      .setName("products")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userFactors.count()
      prodFactors.count()
    }
    new MatrixFactorizationModel(rank, userFactors, prodFactors)
  }


  def run(ratings: JavaRDD[Rating]): MatrixFactorizationModel = run(ratings.rdd)
}


object ALS {

  def train(
      ratings: RDD[Rating],
      rank: Int,
      iterations: Int,
      lambda: Double,
      blocks: Int,
      seed: Long
    ): MatrixFactorizationModel = {
    new ALS2(blocks, blocks, rank, iterations, lambda, false, 1.0, seed).run(ratings)
  }


  def train(
      ratings: RDD[Rating],
      rank: Int,
      iterations: Int,
      lambda: Double,
      blocks: Int
    ): MatrixFactorizationModel = {
    new ALS2(blocks, blocks, rank, iterations, lambda, false, 1.0).run(ratings)
  }


  def train(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double)
    : MatrixFactorizationModel = {
    train(ratings, rank, iterations, lambda, -1)
  }


  def train(ratings: RDD[Rating], rank: Int, iterations: Int)
    : MatrixFactorizationModel = {
    train(ratings, rank, iterations, 0.01, -1)
  }


  def trainImplicit(
      ratings: RDD[Rating],
      rank: Int,
      iterations: Int,
      lambda: Double,
      blocks: Int,
      alpha: Double,
      seed: Long
    ): MatrixFactorizationModel = {
    new ALS2(blocks, blocks, rank, iterations, lambda, true, alpha, seed).run(ratings)
  }


  def trainImplicit(
      ratings: RDD[Rating],
      rank: Int,
      iterations: Int,
      lambda: Double,
      blocks: Int,
      alpha: Double
    ): MatrixFactorizationModel = {
    new ALS2(blocks, blocks, rank, iterations, lambda, true, alpha).run(ratings)
  }


  def trainImplicit(ratings: RDD[Rating], rank: Int, iterations: Int, lambda: Double, alpha: Double)
    : MatrixFactorizationModel = {
    trainImplicit(ratings, rank, iterations, lambda, -1, alpha)
  }


  def trainImplicit(ratings: RDD[Rating], rank: Int, iterations: Int)
    : MatrixFactorizationModel = {
    trainImplicit(ratings, rank, iterations, 0.01, -1, 1.0)
  }
}