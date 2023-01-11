package org.apache.spark.graphx.lotan.graphpropbase
import scala.collection.mutable.{Map => mMap}
import constants.Constants
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import constants.Types
import constants.TypesMKII
import constants.VertexProperty
import constants.VertexPropertyMKII
import constants.VertexPropertyBase
import constants.EdgeProperty
import scala.math.sqrt
import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer
import java.util.StringTokenizer
import graphgenerators.GraphGenerators
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector

object SendFinished extends Serializable {
    import sys.process._
    lazy val res = "/mnt/nfs/gsys/gsys/pipe.py --send_finished" !
}

object SendTerm extends Serializable {
    import sys.process._
    lazy val res = "/mnt/nfs/gsys/gsys/pipe.py --send_term" !
}

class GraphPropBase(val pargs: mMap[Symbol, Int]) extends Constants {
    var spark: SparkSession = _
    var sc: SparkContext = _
    // type GRAPHTYPE
    // var graph: GRAPHTYPE
    // var originalGraph: GRAPHTYPE
    // var reverseGraph: GRAPHTYPE
    var partitionStrategy: PartitionStrategy = _
    var defaultStorageLevel: StorageLevel = _
    var defaultStorageLevelEdges: StorageLevel = _
    val eachWParts = pargs('numEParts) / pargs('numMachines)
    val eachTCPUs = (pargs('CPUs) / eachWParts).max(1)
    // val numThreads = (pargs('CPUs) * pargs('numMachines))
    var cachedCount = 0
    val cachedIdSet = scala.collection.mutable.SortedSet[Int]()
    if (pargs('isCache) == 1) {
        defaultStorageLevel = StorageLevel.MEMORY_ONLY
    } else {
        // WARNING: GraphX caches graphs anyway but they do it MEMORY_ONLY,
        // overwrite the behavior here
        if (pargs('dataset) == 3) {
            // papers100M
            // defaultStorageLevel = StorageLevel.DISK_ONLY
            defaultStorageLevel = StorageLevel.DISK_ONLY

            defaultStorageLevelEdges = defaultStorageLevel
        } else {
            defaultStorageLevel = StorageLevel.MEMORY_AND_DISK_SER
            defaultStorageLevelEdges = StorageLevel.MEMORY_AND_DISK_SER
        }

    }
    if (pargs('E2D) == 0) {
        partitionStrategy = PartitionStrategy.EdgePartition1D
    } else if (pargs('E2D) == 1) {
        partitionStrategy = PartitionStrategy.EdgePartition2D
    } else if (pargs('E2D) == 2) {
        partitionStrategy = PartitionStrategy.RandomVertexCut
    } else if (pargs('E2D) == 3) {
        partitionStrategy = PartitionStrategy.CanonicalRandomVertexCut
    }
    def getKryoConfigs = {
        var conf = new SparkConf()
        val dummy = DenseVector(1.0f, 2.3f, 3.4f)
        val dummy1 = SparseVector(1.0f, 2.3f, 3.4f)
        val dummy2 = new VertexPropertyMKII(0)
        GraphXUtils.registerKryoClasses(conf)
        conf.registerKryoClasses(
            Array(
                classOf[VertexPropertyMKII],
                classOf[VertexPropertyMKII],
                classOf[Array[VertexPropertyMKII]],
                classOf[constants.VertexPropertyMKII],
                classOf[EdgeProperty],
                classOf[Array[EdgeProperty]],
                classOf[DenseVector[Float]],
                classOf[Array[DenseVector[Object]]],
                classOf[Array[Array[Byte]]],
                classOf[SparseVector[Float]],
                classOf[breeze.collection.mutable.SparseArray[Object]],
                classOf[breeze.collection.mutable.SparseArray[Float]],
                classOf[Array[SparseVector[Object]]],
                classOf[Array[Tuple2[Any, Any]]],
                classOf[Array[Array[Tuple2[Any, Any]]]],
                classOf[Array[org.apache.spark.graphx.Edge[Object]]],
                SendFinished.getClass,
                Class.forName("scala.reflect.ClassTag$GenericClassTag"),
                Class.forName("breeze.linalg.SparseVector$mcF$sp"),
                Class.forName("constants.VertexPropertyMKII"),
                // classOf[org.apache.spark.graphx.impl.VertexAttributeBlock],
                Class.forName(
                    "org.apache.spark.graphx.util.collection.GraphXPrimitiveKeyOpenHashMap$mcJI$sp"
                ),
                Class.forName(
                    "org.apache.spark.graphx.lotan.graphpropbase.SendTerm$"
                ),
                Class.forName("breeze.storage.Zero$FloatZero$"),
                Class.forName("scala.reflect.ManifestFactory$LongManifest"),
                Class.forName("scala.reflect.ManifestFactory$IntManifest"),
                Class.forName("java.lang.invoke.SerializedLambda"),
                Class.forName(
                    "org.apache.spark.util.collection.OpenHashSet$LongHasher"
                ),
                Class.forName(
                    "org.apache.spark.graphx.impl.ShippableVertexPartition"
                ),
                Class.forName(
                    "org.apache.spark.graphx.impl.RoutingTablePartition"
                ),
                // Class.forName("[Lscala.Tuple2"),
                dummy.getClass,
                dummy1.getClass,
                dummy2.getClass,
                
            )
        )
        conf.set(
            "spark.serializer",
            "org.apache.spark.serializer.KryoSerializer"
        )
        conf.set("spark.kryoserializer.buffer", "64k")
        conf.set("spark.kryoserializer.buffer.max", "64m")
        conf.set("spark.kryo.registrationRequired", "true")
        // conf.set("spark.executor.extraJavaOptions", "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+UseG1GC -XX:G1HeapRegionSize=32m")
        // conf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=20")

        conf

    }

    def createSpark(): Unit = {
        val memory = "120G"
        var localityWait = if (pargs('hardPartition) == 1) {
            "0"
        } else {
            "10000000s"
        }
        if (pargs('localityWaitZero) == 1) {
            localityWait = "0"
        }
        spark = SparkSession.builder
            .config(getKryoConfigs)
            .appName("Simple Application")
            .master("spark://10.10.1.1:7077")
            //     .config("spark.executor.instances", "4")
            // .config("spark.driver.memory", "2g")
            .config("spark.task.cpus", eachTCPUs)
            .config("spark.driver.host", "master")
            .config("spark.executor.memory", memory)
            .config("spark.memory.fraction", "0.6")
            .config("spark.network.timeout", "10000000s")
            .config("spark.locality.wait", localityWait)
            // .config("spark.executor.cores", eachWParts)
            .config("spark.default.parallelism", pargs('numEParts))
            // .config("spark.files.maxPartitionBytes", 10000000)
            .config("spark.hadoop.validateOutputSpecs", "False")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "True")
            .getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")
    }
    def cacheGraph[A, B](
        g: Graph[A, B],
        name: String = "graph",
        lease: String = "perm"
    ) = {
        val edgeName = s"EdgeRDD_$name"
        val vertexName = s"VertexRDD_$name"
        this.cacheRDD_(g.edges.partitionsRDD, edgeName, graphCache = true)

        this.cacheRDD_(g.vertices.partitionsRDD, vertexName, graphCache = true)
        
        
        println(g.edges.partitionsRDD.name)
        println(g.vertices.partitionsRDD.name)
        println(g.edges.partitionsRDD.id)
        println(g.vertices.partitionsRDD.id)
        // cachedNamesSet += g.edges.name
        // cachedNamesSet += g.vertices.name
        if (lease == "perm") {
            // WARNING: the EdgeRDD and VertexRDD in graphx are incomplete
            // as they do not implement the necessary id and name methods
            // doing a hacky calculation to get the actual underlying RDD id
            // this.cachedIdSet += g.edges.id - 1
            // this.cachedIdSet += g.vertices.id - 2
            // this.cachedIdSet += g.vertices.id - 1
            this.cachedIdSet += g.edges.partitionsRDD.id
            this.cachedIdSet += g.vertices.partitionsRDD.id

        }
    }
    def unmarkGraph[A, B](g: Graph[A, B], name: String = "graph") = {
        this.cachedIdSet -= g.edges.id - 1
        this.cachedIdSet -= g.vertices.id - 1
    }
    def gc(blocking: Boolean = false) = {
        val cachedMap = this.sc.getPersistentRDDs
        println(cachedMap)
        println(cachedIdSet)
        for ((id, rdd) <- cachedMap) {
            val name = rdd.name
            if (!this.cachedIdSet.contains(id)) {
                rdd.unpersist(blocking)
                // println(s"Unpersist $id")
            }

        }
        println(this.sc.getPersistentRDDs)
        System.gc()
    }

    def cacheRDD_[T](
        rdd: RDD[T],
        name: String = "",
        graphCache: Boolean = false
    ) = {
        val storageLevel = if (graphCache) {
            defaultStorageLevel
        } else {
            // normal RDDs are kept in mem
            StorageLevel.MEMORY_AND_DISK_SER
        }
        rdd.setName(name).persist(storageLevel)

        // remove dependenciees
        // rdd.cleanShuffleDependencies(true)
        rdd.localCheckpoint()

        // trigger
        rdd.count()
    }

    def cacheRDD[T](
        rdd: RDD[T],
        RDDName: String = "RDD",
        lease: String = "epoch",
        id: Int = -1
    ) = {

        val realRDD = rdd match {
            case x: VertexRDD[_] => x.partitionsRDD
            case x: EdgeRDD[_] => x.partitionsRDD
            case _ => rdd
        }

        val realID = if (id != -1){
            id
        } else {
            realRDD.id
        }

        var name = if (lease == "epoch") {
            s"epoch_$cachedCount"
        } else {
            this.cachedIdSet += realID
            if (lease == "tmp") {
                s"tmp_$cachedCount"
            } else {
                s"perm_$cachedCount"
            }

        }
        name = name + "_" + RDDName
        println(s"Caching $realRDD, $name, $realID")
        this.cacheRDD_(realRDD, name, graphCache = false)
        this.cachedCount += 1

    }

    def uncacheRDD[T](rdd: RDD[T], id: Int = -1) = {
        val realRDD = rdd match {
            case x: VertexRDD[_] => x.partitionsRDD
            case x: EdgeRDD[_] => x.partitionsRDD
            case _ => rdd
        }

        val realID = if (id != -1){
            id
        } else {
            realRDD.id
        }
        realRDD.unpersist(true)
        this.cachedIdSet -= realID
    }

    def getReplicationFactor[A, B](plainGraph: Graph[A, B]) = {
        import org.apache.spark.sql.functions._
        val edgeRDD = plainGraph.edges
            .map { e => (e.srcId, e.dstId) }
            .mapPartitionsWithIndex { (partitionID, itr) =>
                itr.toList.map(x => (partitionID, x._1, x._2)).iterator
            }
        val edgeDF = spark
            .createDataFrame(edgeRDD)
            .toDF("partitionID", "srcID", "dstID")
        val srcDF =
            edgeDF.select(col("partitionID"), col("srcID").alias("vid"))
        val dstDF =
            edgeDF.select(col("partitionID"), col("dstID").alias("vid"))
        val vidDF = srcDF.union(dstDF)
        val aggDF = vidDF
            .groupBy("vid")
            .agg(expr("count(distinct partitionID) as replicationFactor"))
        val avgReplicationFactor =
            aggDF.select(mean("replicationFactor")).collect()(0).toString()
        val avgForwardReplicationFactor = srcDF
            .groupBy("vid")
            .agg(expr("count(distinct partitionID) as replicationFactor"))
            .select(mean("replicationFactor"))
            .collect()(0)
            .toString()
        val avgBackpropReplicationFactor = dstDF
            .groupBy("vid")
            .agg(expr("count(distinct partitionID) as replicationFactor"))
            .select(mean("replicationFactor"))
            .collect()(0)
            .toString()
        println(
            s"Average Replication Factor: $avgReplicationFactor, Average Forward Replication Factor: $avgForwardReplicationFactor, Average Backprop Replication Factor: $avgBackpropReplicationFactor"
        )
    }
    def savePlainGraph[VD, ED](
        graph: Graph[VD, ED],
        edgePath: String,
        vertexPath: String
    ) = {
        graph.edges
            .map { e => s"${e.srcId},${e.dstId}" }
            .saveAsTextFile(edgePath)
        graph.vertices
            .map { case (id, _) => s"$id" }
            .saveAsTextFile(vertexPath)
    }
    def selfEdges[T <: VertexPropertyBase](feats: RDD[(Long, T)]) = {
        feats.map { case (id, _) =>
            Edge(id, id, new EdgeProperty())
        }
    }
    def tokenize(command: String): Seq[String] = {
        val buf = new ArrayBuffer[String]
        val tok = new StringTokenizer(command)
        while (tok.hasMoreElements) {
            buf += tok.nextToken()
        }
        buf.toSeq
    }

    def printGraph[VD](g: Graph[VD, EdgeProperty]) = {
        // var tmpGraph: GraphType = null
        // if (pargs('noReverseGraph) == 1) {
        //     tmpGraph = graph
        // } else {
        //     tmpGraph = reverseGraph
        // }
        // trigger action
        val vcount = g.vertices.count()
        val ecount = g.edges.count()
        println(s"#vertices: $vcount, #edges: $ecount")
        val samples = g.vertices.take(10)
        for ((id, vprop) <- samples) {
            val arr_str = vprop.toString()
            println(s"ID: $id, Content: $arr_str")
        }
        val edgeSamples = g.edges.take(10)
        for (eprop <- edgeSamples) {
            println(
                s"Src: ${eprop.srcId}, Dst: ${eprop.dstId}, Weight: ${eprop.attr.weight}"
            )
        }

    }
    def sendFinished() = {
        // stupidly large RDD just to force Spark to distribute this to ALL executors
        val controlRDD = sc.parallelize(Array(0, 320), 320)
        val pm = sc.broadcast(SendFinished)
        val res = controlRDD
            .mapPartitions(itr => {
                pm.value.res
                // Other stuff here.
                itr
            })
            .count()

    }

    def sendTerm() = {
        // stupidly large RDD just to force Spark to distribute this to ALL executors
        val controlRDD = sc.parallelize(Array(0, 320), 320)
        val pm = sc.broadcast(SendTerm)
        val res = controlRDD
            .mapPartitions(itr => {
                pm.value.res
                // Other stuff here.
                itr
            })
            .count()

    }

}
