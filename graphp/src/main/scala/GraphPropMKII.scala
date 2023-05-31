// Copyright 2023 Yuhao Zhang and Arun Kumar. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

package org.apache.spark.graphx.lotan
import scala.collection.mutable.{Map => mMap}
import constants.Constants
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import graphgenerators.GraphGenerators
import graphpropbase.GraphPropBase
import constants.Types
import constants.TypesMKII
// import constants.VertexProperty
import constants.{VertexPropertyMKII => VertexProperty}
import constants.VertexPropertyBase
import constants.EdgeProperty
import scala.math.sqrt
import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer
import java.util.StringTokenizer
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import upickle.default._
import com.google.common.collect.Lists

sealed trait OneOf[A, B, C, D]
case class First[A, B, C, D](a: A) extends OneOf[A, B, C, D]
case class Second[A, B, C, D](b: B) extends OneOf[A, B, C, D]
case class Third[A, B, C, D](c: C) extends OneOf[A, B, C, D]
case class Fourth[A, B, C, D](c: D) extends OneOf[A, B, C, D]

// trait Embs[A]{
//     def getString(x: A): String
// }

// class DenseEmbs[A] (val vec: DenseVector[A]) extends Embs {
// }
// class SparseEmbs[A] (val vec: SparseVector[A]) extends Embs {
// }
class AffinityRDD[T: ClassTag](
    var prev: RDD[T],
    hosts: Seq[String],
    partitionerOverride: Partitioner = null
) extends RDD[T](prev) {

    override val partitioner = partitionerOverride match {
        case null => firstParent[T].partitioner
        case _    => Some(partitionerOverride)
    }

    override def getPartitions: Array[Partition] = firstParent[T].partitions
    override def compute(
        split: Partition,
        context: TaskContext
    ): Iterator[T] = {
        firstParent[T].iterator(split, context)
    }
    override def clearDependencies(): Unit = {
        super.clearDependencies()
        prev = null
    }
    override protected def getPreferredLocations(
        split: Partition
    ): Seq[String] = {
        Seq(hosts(split.index % hosts.length))
    }
    def splash() = {
        for (i <- firstParent[T].partitions) {
            println(this.getPreferredLocations(i))
        }

    }

}

class GraphPropMKII(
    override val pargs: mMap[Symbol, Int]
) extends GraphPropBase(pargs)
    with TypesMKII {

    var graph: GraphType = _
    var originalGraph: GraphType = _
    var originalReverseGraph: GraphType = _
    var reverseGraph: GraphType = _
    val datasetDir = datasetBetterMap(pargs('dataset))
    val base = "hdfs://master:9000/"
    val verticesFile = base + "vertices"
    val edgesFile = base + "edges"
    val edgesRevFile = base + "edgesRev"

    def hardPartition(): Unit = {
        // add a deadweight
        val vertices = this.graph.vertices.map(f =>
            (
                f._1,
                new VertexProperty(
                    num = f._2.num,
                    emb = f._2.emb,
                    // grad = DenseVector.fill(500, 0f),
                    feat = f._2.feat,
                    vecSparse = f._2.vecSparse
                )
            )
        )
        val edges = this.graph.edges
        val edgesRev = this.reverseGraph.edges
        vertices
            // .partitionBy(new HashPartitioner(pargs('numVParts)))
            .saveAsObjectFile(verticesFile)
        edges.saveAsObjectFile(edgesFile)
        edgesRev.saveAsObjectFile(edgesRevFile)
    }

    def loadFeats(featFile: String) = {
        val rdd = spark.sparkContext
            .textFile(featFile, pargs('numVParts))
            .map(
                if (pargs('sparse) == 1) { f =>
                    {
                        val splited = f.split(",")
                        val id = splited(0).toLong
                        // var labeled = 1.0
                        // var feat: Array[String] = null
                        // if (pargs('labeledInfo) == 1){
                        //     labeled = splited(1).toDouble
                        //     feat = splited.drop(2)
                        // } else {
                        //     feat = splited.drop(1)
                        // }
                        val labeled = 1.0
                        val feat = splited.drop(1)
                        
                        val featCasted = feat.map(x => x.toFloat)
                        val dfeat = DenseVector(featCasted)
                        val keys = (dfeat :!= 0.0f)
                        val values = dfeat(keys)
                        (
                            id,
                            new VertexProperty(
                                num = id,
                                vecSparse = new SparseVector(
                                    keys.activeKeysIterator.toArray,
                                    values.toArray,
                                    dfeat.length
                                ),
                                labeled = labeled
                            )
                        )

                    }

                } else { f =>
                    {
                        val splited = f.split(",")
                        val id = splited(0).toLong
                        // var labeled = 1.0
                        // var feat: Array[String] = null
                        // if (pargs('labeledInfo) == 1){
                        //     labeled = splited(1).toDouble
                        //     feat = splited.drop(2)
                        // } else {
                        //     feat = splited.drop(1)
                        // }
                        val labeled = 1.0
                        val feat = splited.drop(1)

                        val featCasted = feat.map(x => x.toFloat)
                        (
                            id,
                            new VertexProperty(
                                num = id,
                                feat = DenseVector(featCasted),
                                labeled = labeled
                            )
                        )

                    }

                }
            )
            .partitionBy(new HashPartitioner(pargs('numVParts)))
        // .cache()
        // .repartition(pargs('numEParts)).cache()
        // rdd.count()
        // rdd.take(10).foreach(println)
        rdd

    }

    def loadGraphCSV() = {
        val edgesFile = datasetDir + "/" + EDGES
        val metaFile = datasetDir + "/" + META
        val featFile = datasetDir + "/" + FEATURES

        val reverseEdges = spark.sparkContext
            .textFile(edgesFile)
            .map(f => {
                val splited = f.split(",")
                Edge(splited(1).toLong, splited(0).toLong, new EdgeProperty())
            })
            .repartition(pargs('numEParts))

        val feats = this.loadFeats(featFile)

        val finEdges = if (pargs('selfLoop) == 1) {
            reverseEdges.union(this.selfEdges(feats))
        } else {
            reverseEdges
        }

        this.reverseGraph = Graph(
            feats,
            finEdges,
            null,
            edgeStorageLevel = defaultStorageLevelEdges,
            vertexStorageLevel = defaultStorageLevel
        )

        // val inDegrees = this.reverseGraph.aggregateMessages[Double](
        //     triplet => triplet.sendToDst(triplet.srcAttr.labeled), _ + _, TripletFields.Src)

        // val outDegrees = this.reverseGraph.aggregateMessages[Double](
        //     triplet => triplet.sendToSrc(triplet.dstAttr.labeled), _ + _, TripletFields.Dst)

        val inDegrees = this.reverseGraph.inDegrees
        val outDegrees = this.reverseGraph.outDegrees

        this.reverseGraph = this.reverseGraph
            .joinVertices(inDegrees)((id, vprop, degrees) => {
                val inD = degrees.toDouble.max(1)
                new VertexProperty(
                    num = vprop.num,
                    feat = vprop.feat,
                    inDegrees = inD,
                    vecSparse = vprop.vecSparse
                )
            })
            .joinVertices(outDegrees)((id, vprop, degrees) => {
                val outD = degrees.toDouble.max(1)
                new VertexProperty(
                    num = vprop.num,
                    feat = vprop.feat,
                    inDegrees = vprop.inDegrees,
                    outDegrees = outD,
                    vecSparse = vprop.vecSparse
                )
            })
        if (pargs('normalize) == 1) {
            // symmetrical norm
            this.reverseGraph = this.reverseGraph.mapTriplets(triplet =>
                new EdgeProperty(
                    weight = (1.0 / (sqrt(triplet.dstAttr.outDegrees) * sqrt(
                        triplet.srcAttr.inDegrees
                    ))).toFloat
                )
            )
        } else if (pargs('normalize) == 2) {
            // right norm, to divide the aggregated messages by each node’s in-degrees, which is equivalent to averaging the received messages.
            this.reverseGraph = this.reverseGraph.mapTriplets(triplet =>
                new EdgeProperty(
                    weight = (1.0 /
                        triplet.srcAttr.inDegrees).toFloat
                )
            )
        } else if (pargs('normalize) == 3) {
            // left norm
            this.reverseGraph = this.reverseGraph.mapTriplets(triplet =>
                new EdgeProperty(
                    weight = (1.0 /
                        triplet.dstAttr.outDegrees).toFloat
                )
            )
        }

        this.reverseGraph =
            this.reverseGraph.partitionBy(partitionStrategy, pargs('numEParts))

        this.graph = reverseGraph.reverse.partitionBy(
            partitionStrategy,
            pargs('numEParts)
        )

    }
    def loadGraphOBJ() = {

        val verticesNoAFF = sc
            .objectFile[(Long, VertexProperty)](
                verticesFile
                // minPartitions = pargs('numVParts)
            )
        val vertices =
            new AffinityRDD(
                verticesNoAFF,
                hosts,
                new HashPartitioner(pargs('numVParts))
            ).cache()

        // vertices.splash()
        val edgesNoAFF = sc
            .objectFile[Edge[EdgeProperty]](
                edgesFile
                // minPartitions = pargs('numEParts)
            )
        // .sample(false, 0.01).union(this.selfEdges(vertices))
        val edges =
            new AffinityRDD(edgesNoAFF, hosts)

        // vertices.count()
        // edges.count()

        // sc.makeRDD()
        this.graph = Graph(
            vertices,
            edges,
            null,
            edgeStorageLevel = defaultStorageLevelEdges,
            vertexStorageLevel = defaultStorageLevel
        )

        if (pargs('noReverseGraph) == 0) {
            val edgesRevNoAFF = sc.objectFile[Edge[EdgeProperty]](
                edgesRevFile
                // minPartitions = pargs('numEParts)
            )
            val edgesRev =
                new AffinityRDD(edgesRevNoAFF, hosts)

            // edgesRev.count()

            this.reverseGraph = Graph(
                vertices,
                edgesRev,
                null,
                edgeStorageLevel = defaultStorageLevelEdges,
                vertexStorageLevel = defaultStorageLevel
            )
        } else {
            this.reverseGraph = this.graph
        }

    }

    def loadGraph() = {

        if (pargs('hardPartitionRead) == 1) {
            this.loadGraphOBJ()
            this.cacheGraph(this.graph, "origraph")
            if (pargs('noReverseGraph) == 0) {
                this.cacheGraph(this.reverseGraph, "orireversegraph")
            }
        } else {
            this.loadGraphCSV()
        }

        this.originalGraph = this.graph
        this.originalReverseGraph = this.reverseGraph
        if (pargs('getReplication) == 1) {
            getReplicationFactor(graph)
        }

        // if (pargs('fillFeatures) == 1) {
        //     loadGraphFeatures(plainGraph, reversePlainGraph)
        // }
    }

    // def loadGraphFeatures(
    //     plainGraph: Graph[Long, Int],
    //     reversePlainGraph: Graph[Long, Int]
    // ): GraphType = {
    //     val featFile = datasetDir + "/" + FEATURES
    //     val feats = spark.sparkContext
    //         .textFile(featFile)
    //         .map(f => {
    //             val splited = f.split(",")
    //             val id = splited(0).toLong
    //             val feat = splited.drop(1)
    //             val featCasted = feat.map(x => x.toFloat)

    //             (id, featCasted)
    //         })

    //     this.graph = plainGraph
    //         .outerJoinVertices(feats)((id, _, feat) => {
    //             feat match {
    //                 case Some(feat) => {
    //                     if (this.pargs('sparse) == 1){
    //                         val dfeat = DenseVector(feat)
    //                         val keys = (dfeat :!= 0.0f)
    //                         val values = dfeat(keys)
    //                         new VertexProperty(num = id, vecSparse = new SparseVector(keys.activeKeysIterator.toArray, values.toArray, dfeat.length))

    //                     }
    //                     else{
    //                         new VertexProperty(num = id, feat = DenseVector(feat))
    //                     }
    //                 }
    //                 case _ => throw new Exception("Unexpected data")
    //             }
    //         })
    //         .mapEdges(e => new EdgeProperty())

    //     this.reverseGraph = reversePlainGraph
    //         .mapVertices((id, _) => new VertexProperty(id))
    //         .mapEdges(e => new EdgeProperty())
    //         .cache()
    //     this.cacheGraph(this.graph, "origraph")
    //     this.cacheGraph(this.reverseGraph, "orireversegraph")

    //     this.originalGraph = this.graph
    //     return this.graph
    // }
    def updateEmbs(updatedEmbs: Either[EmbsType, EmbsSparseType]) = {
        updatedEmbs match {
            case Left(x) => {
                this.graph = this.graph
                    .joinVertices(x)((id, vprop, updatedEmb) => {
                        // vprop.messageArray = message
                        new VertexProperty(num = vprop.num, emb = updatedEmb)
                    })
            }
            case Right(x) => {
                this.graph = this.graph
                    .joinVertices(x)((id, vprop, updatedEmb) => {
                        new VertexProperty(
                            num = vprop.num,
                            vecSparse = updatedEmb
                        )
                    })
            }
        }
        // always on the forward graph

    }

    def gatherScatter(layerIDX: Int) = {
        _gatherScatter(this.graph, layerIDX)
    }

    def _gatherScatter(g: GraphType, layerIDX: Int): MessageType = {
        def funcNormal(
            triplet: EdgeContext[VertexProperty, EdgeProperty, GatherArrayType]
        ): Unit = { // Map Function
            // Send message to destination vertex containing
            triplet.sendToDst(
                Array(
                    (triplet.srcId, triplet.srcAttr.emb * triplet.attr.weight)
                )
            )
        }
        def funcFirst(
            triplet: EdgeContext[VertexProperty, EdgeProperty, GatherArrayType]
        ): Unit = { // Map Function
            // Send message to destination vertex containing
            triplet.sendToDst(
                Array(
                    (triplet.srcId, triplet.srcAttr.feat * triplet.attr.weight)
                )
            )
        }

        val func = if (layerIDX == 0) {
            funcFirst(_)
        } else {
            funcNormal(_)
        }

        val messages = g.aggregateMessages[GatherArrayType](
            func,
            (
                a,
                b
            ) => a ++ b,
            tripletFields = TripletFields.Src
        )
        return messages
    }
    def _gatherScatterSumAggSparse(
        g: GraphType,
        layerIDX: Int
    ): EmbsSparseType = {
        def funcNormalSparse(
            triplet: EdgeContext[VertexProperty, EdgeProperty, EmbSparseType]
        ): Unit = {
            triplet.sendToDst(triplet.srcAttr.vecSparse * triplet.attr.weight)
        }

        val func = funcNormalSparse(_)
        val messages = g.aggregateMessages[EmbSparseType](
            func,
            (
                a,
                b
            ) => a + b,
            tripletFields = TripletFields.Src
        )

        return messages
    }
    def _gatherScatterSumAgg(
        g: GraphType,
        layerIDX: Int
    ): EmbsType = {
        def funcNormal(
            triplet: EdgeContext[VertexProperty, EdgeProperty, EmbType]
        ): Unit = {
            triplet.sendToDst(triplet.srcAttr.emb * triplet.attr.weight)
        }
        def funcFirst(
            triplet: EdgeContext[VertexProperty, EdgeProperty, EmbType]
        ): Unit = {
            triplet.sendToDst(triplet.srcAttr.feat * triplet.attr.weight)
        }

        val func = if (layerIDX == 0) {
            funcFirst(_)
        } else {
            funcNormal(_)
        }

        val messages = g.aggregateMessages[EmbType](
            func,
            (
                a,
                b
            ) => a + b,
            tripletFields = TripletFields.Src
        )

        return messages
    }
    def genRandomGraph() = {
        var plainGraph = GraphGenerators
            .logNormalGraph(
                sc,
                numVertices = pargs('numVertices).asInstanceOf[Int],
                numEParts = pargs('numEParts).asInstanceOf[Int],
                seed = SEED,
                edgeStorageLevel = defaultStorageLevelEdges,
                vertexStorageLevel = defaultStorageLevel
            )
        val selfEdges = plainGraph.vertices.map { case (id, v) =>
            Edge(id, id, 1)
        }

        // for (e <- selfEdges.take(10)) {
        //     //     val arr_str = vprop.emb.mkString(", ")
        //         val ids = (e.srcId, e.dstId)
        //         println(s"ID: $ids")
        //     }
        // for (e <- newEdges.take(10)) {
        //     //     val arr_str = vprop.emb.mkString(", ")
        //         val ids = (e.srcId, e.dstId)
        //         println(s"ID: $ids")
        // }
        plainGraph = Graph(
            plainGraph.vertices,
            plainGraph.edges.union(selfEdges),
            0,
            edgeStorageLevel = defaultStorageLevelEdges,
            vertexStorageLevel = defaultStorageLevel
        )
        plainGraph =
            plainGraph.partitionBy(partitionStrategy, pargs('numEParts))

        val reversePlainGraph =
            plainGraph.reverse.partitionBy(partitionStrategy, pargs('numEParts))
        if (pargs('savePlain) == 1) {
            savePlainGraph(plainGraph, EDGE_FILE_PATH, VERTEX_FILE_PATH)
            savePlainGraph(
                reversePlainGraph,
                REVERSE_EDGE_FILE_PATH,
                REVERSE_VERTEX_FILE_PATH
            )
        }
        if (pargs('getReplication) == 1) {
            getReplicationFactor(plainGraph)
        }

        if (pargs('fillFeatures) == 1) {
            fillGraphFeatures(plainGraph, reversePlainGraph)
        }
    }

    def fillGraphFeatures(
        plainGraph: Graph[Long, Int],
        reversePlainGraph: Graph[Long, Int]
    ): GraphType = {
        this.graph = plainGraph
            .mapVertices((id, _) =>
                new VertexProperty(id, feat = DenseVector(Array.fill(100)(1f)))
            )
            .mapEdges(e => new EdgeProperty())
            .cache()
        this.reverseGraph = reversePlainGraph
            .mapVertices((id, _) => new VertexProperty(id))
            .mapEdges(e => new EdgeProperty())
            .cache()

        // if (pargs('isCache) == 1) {
        //     cacheGraph(graph)
        //     cacheGraph(reverseGraph)
        // } else {
        //     // WARNING: GraphX caches graphs anyway but they do it MEMORY_ONLY,
        //     // overwrite the behavior here
        //     persistGraph(graph)
        //     persistGraph(reverseGraph)
        // }
        this.originalGraph = this.graph
        return this.graph
    }
    def gatherScatterSumAgg(layerIDX: Int): Either[EmbsType, EmbsSparseType] = {
        if ((this.pargs('sparse) == 1)) {
            Right(_gatherScatterSumAggSparse(this.graph, layerIDX))
        } else {
            Left(_gatherScatterSumAgg(this.graph, layerIDX))
        }

    }

    def backpropGatherScatterSingleSum()
        : Either[VertexRDD[GradType], VertexRDD[GradSparseType]] = {
        if (pargs('noReverseGraph) == 1) {
            backpropGatherScatterNormalSingleSum()
        } else {
            backpropGatherScatterReverseSingleSum()
        }
    }

    def backpropGatherScatterNormalSingleSum()
        : Either[VertexRDD[GradType], VertexRDD[GradSparseType]] = {
        val g = this.graph
        val fields = TripletFields.Dst

        if (this.pargs('sparse) == 1) {
            def mapFunc(
                triplet: EdgeContext[
                    VertexProperty,
                    EdgeProperty,
                    GradSparseType
                ]
            ): Unit = {
                triplet.sendToSrc(
                    triplet.dstAttr.vecSparse * triplet.attr.weight
                )
            }

            //
            def reduceFunc(
                a: GradSparseType,
                b: GradSparseType
            ) = {
                a + b
            }

            Right(
                this._backpropGatherScatter[GradSparseType](
                    g,
                    mapFunc,
                    reduceFunc,
                    fields
                )
            )
        } else {
            def mapFunc(
                triplet: EdgeContext[VertexProperty, EdgeProperty, GradType]
            ): Unit = {
                triplet.sendToSrc(triplet.dstAttr.grad * triplet.attr.weight)
            }

            //
            def reduceFunc(
                a: GradType,
                b: GradType
            ) = {
                a + b
            }

            Left(
                this._backpropGatherScatter[GradType](
                    g,
                    mapFunc,
                    reduceFunc,
                    fields
                )
            )

        }

    }

    def backpropGatherScatterReverseSingleSum()
        : Either[VertexRDD[GradType], VertexRDD[GradSparseType]] = {
        val g = this.reverseGraph
        val fields = TripletFields.Src

        if (this.pargs('sparse) == 1) {
            def mapFunc(
                triplet: EdgeContext[
                    VertexProperty,
                    EdgeProperty,
                    GradSparseType
                ]
            ): Unit = {
                triplet.sendToDst(
                    triplet.srcAttr.vecSparse * triplet.attr.weight
                )
            }
            def reduceFunc(
                a: GradSparseType,
                b: GradSparseType
            ) = {
                a + b
            }
            Right(
                this._backpropGatherScatter[GradSparseType](
                    g,
                    mapFunc,
                    reduceFunc,
                    fields
                )
            )

        } else {
            def mapFunc(
                triplet: EdgeContext[VertexProperty, EdgeProperty, GradType]
            ): Unit = {
                triplet.sendToDst(triplet.srcAttr.grad * triplet.attr.weight)
            }

            //
            def reduceFunc(
                a: GradType,
                b: GradType
            ) = {
                a + b
            }

            Left(
                this._backpropGatherScatter[GradType](
                    g,
                    mapFunc,
                    reduceFunc,
                    fields
                )
            )

        }

    }

    def _backpropGatherScatter[T: ClassTag](
        g: GraphType,
        mapFn: EdgeContext[VertexProperty, EdgeProperty, T] => Unit,
        reduceFn: (T, T) => T,
        tripletFields: TripletFields
    ) = {
        g.aggregateMessages[T](
            mapFn,
            // Add counter and age
            reduceFn,
            tripletFields = tripletFields
        )
    }
    // def updateGrads(
    //     gradsMap: OneOf[
    //         MapOfGradsType,
    //         GradsType,
    //         MapOfGradsSparseType,
    //         GradsSparseType
    //     ]
    // ) = {
    //     updateGraph(gradsMap)
    // }

    def updateGraph(
        messages: OneOf[
            MapOfGradsType,
            GradsType,
            MapOfGradsSparseType,
            GradsSparseType
        ]
    ) = {

        var tmpGraph = this.getGraphToUpdate()
        tmpGraph = messages match {
            case First(x) =>
                tmpGraph
                    .joinVertices(x)((id, vprop, message) => {
                        // vprop.messageArray = message
                        new VertexProperty(
                            num = vprop.num,
                            messageMap = message.toMap
                        )
                    })
            case Second(x) =>
                tmpGraph
                    .joinVertices(x)((id, vprop, message) => {
                        // vprop.messageArray = message
                        new VertexProperty(num = vprop.num, grad = message)
                    })
            // not implemented!
            case Third(x) => tmpGraph
            case Fourth(x) =>
                tmpGraph
                    .joinVertices(x)((id, vprop, message) => {
                        // vprop.messageArray = message
                        new VertexProperty(num = vprop.num, vecSparse = message)
                    })

        }
        this.updateInternalGraph(tmpGraph)
    }

    def updateInternalGraph(tmpGraph: GraphType) = {
        if (pargs('noReverseGraph) == 1) {
            this.graph = tmpGraph
        } else {
            this.reverseGraph = tmpGraph
        }
    }

    def backpropGatherScatter(): EmbsType = {
        var gradientRDD: EmbsType = null
        if (pargs('noReverseGraph) == 1) {
            gradientRDD = backpropGatherScatterNormal()
        } else {
            gradientRDD = backpropGatherScatterReverse()
        }
        return gradientRDD
    }

    def getGraphToUpdate() = {
        if (pargs('noReverseGraph) == 1) {
            graph
        } else {
            reverseGraph
        }
    }

    def backpropGatherScatterNormal(): EmbsType = {

        graph.aggregateMessages[EmbType](
            triplet => { // Map Function
                if (triplet.dstAttr.messageMap != null) {
                    triplet.sendToSrc(
                        triplet.dstAttr.messageMap(
                            triplet.srcId
                        ) * triplet.attr.weight
                    )
                }
            },
            (
                a,
                b
            ) => a + b,
            tripletFields = TripletFields.Dst
        )
    }

    def backpropGatherScatterReverse(): EmbsType = {
        // Aggregate message way
        // val revereGraph = graph.reverse.partitionBy(PARTITION_STRATEGY)

        reverseGraph.aggregateMessages[EmbType](
            triplet => { // Map Function
                // Send message to destination vertex containing
                if ((triplet.srcAttr.messageMap != null)) {
                    triplet.sendToDst(
                        triplet.srcAttr.messageMap(
                            triplet.dstId
                        ) * triplet.attr.weight
                    )
                }
            },
            // Add counter and age
            (
                a,
                b
            ) => a + b,
            tripletFields = TripletFields.Src
        )
    }

    def printRDDElement(
        record: (VertexId, GatherArrayType),
        f: String => Unit
    ) = {

        val jsonString = write(
            (
                record._1,
                record._2.map { case (idx, emb) => (idx, emb.toArray) }
            )
        )
        f(jsonString)
    }

    def printRDDElementGrad(
        record: (VertexId, GradType),
        f: String => Unit
    ) = {
        val jsonString = write((record._1, record._2.toArray))
        f(jsonString)
    }

    def resetCache(blocking: Boolean = false) = {

        this.graph = this.originalGraph
        this.reverseGraph = this.originalReverseGraph
        this.gc(blocking)
    }

}

class GraphPropSparse(
    override val pargs: mMap[Symbol, Int]
) extends GraphPropMKII(pargs) {}
