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
import constants.VertexProperty
import constants.VertexPropertyMKII
import constants.VertexPropertyBase
import constants.EdgeProperty
import scala.math.sqrt
import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer
import java.util.StringTokenizer
import breeze.linalg.DenseVector



class GraphProp(
    override val pargs: mMap[Symbol, Int]
) extends GraphPropBase(pargs)
    with Types {
    type GRAPHTYPE = GraphType
    var graph: GraphType = _
    var originalGraph: GraphType = _
    var reverseGraph: GraphType = _
    def cache() = {
        cacheGraph(graph)
        cacheGraph(reverseGraph)
    }

    def persistGraph[A, B](g: Graph[A, B]) = {
        g.persist(defaultStorageLevel)
    }
    def loadFeats(featFile: String) = {
        spark.sparkContext
            .textFile(featFile)
            .map(f => {
                val splited = f.split(",")
                val id = splited(0).toLong
                val feat = splited.drop(1)
                val featCasted = feat.map(x => x.toFloat)

                (id, new VertexProperty(num = id, feat = featCasted))
            })
            .partitionBy(new HashPartitioner(pargs('numVParts)))
    }

    def loadGraph() = {
        val edgesFile = datasetMap("products") + "/" + EDGES
        val metaFile = datasetMap("products") + "/" + META
        val featFile = datasetMap("products") + "/" + FEATURES

        val reverseEdges = spark.sparkContext
            .textFile(edgesFile)
            .map(f => {
                val splited = f.split(",")
                Edge(splited(1).toLong, splited(0).toLong, new EdgeProperty())
            })

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
            edgeStorageLevel = defaultStorageLevel,
            vertexStorageLevel = defaultStorageLevel
        ).cache()

        this.reverseGraph = this.reverseGraph
            .joinVertices(this.reverseGraph.outDegrees)(
                (id, vprop, degrees) => {
                    new VertexProperty(
                        num = vprop.num,
                        feat = vprop.feat,
                        inDegrees = degrees.toDouble
                    )
                }
            )
            .joinVertices(this.reverseGraph.inDegrees)((id, vprop, degrees) => {
                new VertexProperty(
                    num = vprop.num,
                    feat = vprop.feat,
                    inDegrees = vprop.inDegrees,
                    outDegrees = degrees.toDouble
                )
            })
            .mapTriplets(triplet =>
                new EdgeProperty(
                    weight = (1.0 / (sqrt(triplet.dstAttr.outDegrees) * sqrt(
                        triplet.srcAttr.inDegrees
                    ))).toFloat
                )
            )
            .partitionBy(partitionStrategy, pargs('numEParts))
        this.graph = reverseGraph.reverse.partitionBy(partitionStrategy, pargs('numEParts))
        this.cacheGraph(this.graph, "origraph")
        this.cacheGraph(this.reverseGraph, "orireversegraph")

        this.originalGraph = this.graph

        // val vertices = feats.mapValues(f => {
        //     new VertexProperty(num=f.num)
        // })

        // val vcount = vertices.count()
        // println(s"LOAD VERTICES: $vcount")

        // val plainGraph = Graph[Long, Int](
        //     vertices,
        //     edges,
        //     0,
        //     edgeStorageLevel = defaultStorageLevel,
        //     vertexStorageLevel = defaultStorageLevel
        // ).partitionBy(partitionStrategy, pargs('numEParts))

        // val reversePlainGraph =
        //     plainGraph.reverse
        //         .partitionBy(partitionStrategy, pargs('numEParts))

        // if (pargs('savePlain) == 1) {
        //     savePlainGraph(plainGraph, EDGE_FILE_PATH, VERTEX_FILE_PATH)
        //     savePlainGraph(
        //         reversePlainGraph,
        //         REVERSE_EDGE_FILE_PATH,
        //         REVERSE_VERTEX_FILE_PATH
        //     )
        // }
        if (pargs('getReplication) == 1) {
            getReplicationFactor(graph)
        }

        // if (pargs('fillFeatures) == 1) {
        //     loadGraphFeatures(plainGraph, reversePlainGraph)
        // }
    }

    def loadGraphFeatures(
        plainGraph: Graph[Long, Int],
        reversePlainGraph: Graph[Long, Int]
    ): GraphType = {
        val featFile = datasetMap("products") + "/" + FEATURES
        val feats = spark.sparkContext
            .textFile(featFile)
            .map(f => {
                val splited = f.split(",")
                val id = splited(0).toLong
                val feat = splited.drop(1)
                val featCasted = feat.map(x => x.toFloat)

                (id, featCasted)
            })
        this.graph = plainGraph
            .outerJoinVertices(feats)((id, _, feat) => {
                feat match {
                    case Some(feat) => new VertexProperty(num = id, feat = feat);
                    case _ => throw new Exception("Unexpected data")
                }
            })
            .mapEdges(e => new EdgeProperty())

        this.reverseGraph = reversePlainGraph
            .mapVertices((id, _) => new VertexProperty(id))
            .mapEdges(e => new EdgeProperty())
            .cache()
        this.cacheGraph(this.graph, "origraph")
        this.cacheGraph(this.reverseGraph, "orireversegraph")

        this.originalGraph = this.graph
        return this.graph
    }

    def genRandomGraph() = {
        var plainGraph = GraphGenerators
            .logNormalGraph(
                sc,
                numVertices = pargs('numVertices).asInstanceOf[Int],
                numEParts = pargs('numEParts).asInstanceOf[Int],
                seed = SEED,
                edgeStorageLevel = defaultStorageLevel,
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
            edgeStorageLevel = defaultStorageLevel,
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
                new VertexProperty(id, feat = Array.fill(100)(1))
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

    def _gatherScatter(g: GraphType, layerIDX: Int): MessageType = {
        def funcNormal(
            triplet: EdgeContext[VertexProperty, EdgeProperty, GatherArrayType]
        ): Unit = { // Map Function
            // Send message to destination vertex containing
            triplet.sendToDst(Array((triplet.srcId, triplet.srcAttr.emb)))
        }
        def funcFirst(
            triplet: EdgeContext[VertexProperty, EdgeProperty, GatherArrayType]
        ): Unit = { // Map Function
            // Send message to destination vertex containing
            triplet.sendToDst(Array((triplet.srcId, triplet.srcAttr.feat)))
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

    def _gatherScatterSumAgg(g: GraphType, layerIDX: Int): EmbsType = {
        def funcNormal(
            triplet: EdgeContext[VertexProperty, EdgeProperty, EmbType]
        ): Unit = {
            triplet.sendToDst(triplet.srcAttr.emb.map(_ * triplet.attr.weight))
        }
        def funcFirst(
            triplet: EdgeContext[VertexProperty, EdgeProperty, EmbType]
        ): Unit = {
            triplet.sendToDst(triplet.srcAttr.feat.map(_ * triplet.attr.weight))
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
            ) => (a, b).zipped.map(_ + _),
            tripletFields = TripletFields.Src
        )
        return messages
    }

    // def scatterGatherBack[VD, ED](g: Graph[VD, ED]) = {
    //     g.aggregateMessages[GradType](
    //         triplet => { // Map Function
    //             // Send message to destination vertex containing
    //             triplet.sendToDst(Array((triplet.srcId, triplet.srcAttr.emb)))
    //         },
    //         (
    //             a,
    //             b
    //         ) => a ++ b,
    //         tripletFields = TripletFields.Src
    //     )
    // }

    def gatherScatter(layerIDX: Int) = {
        _gatherScatter(this.graph, layerIDX)
    }
    def gatherScatterSumAgg(layerIDX: Int) = {
        _gatherScatterSumAgg(this.graph, layerIDX)
    }

    def backpropGatherScatter(): MessageType = {
        var gradientRDD: MessageType = null
        if (pargs('noReverseGraph) == 1) {
            gradientRDD = backpropGatherScatterNormal()
        } else {
            gradientRDD = backpropGatherScatterMap()
        }
        return gradientRDD
    }

    def backpropGatherScatterSum(): GradsType = {
        var gradientRDD: GradsType = null
        if (pargs('noReverseGraph) == 1) {
            gradientRDD = this.backpropGatherScatterNormalSum()
        } else {
            gradientRDD = this.backpropGatherScatterMapSum()
        }
        return gradientRDD
    }

    def backpropGatherScatterNormal(): MessageType = {

        graph.aggregateMessages[GatherArrayType](
            triplet => { // Map Function
                if (triplet.dstAttr.messageMap != null) {
                    triplet.sendToSrc(
                        Array(
                            (
                                triplet.dstId,
                                triplet.dstAttr.messageMap(triplet.srcId)
                            )
                        )
                    )
                }
            },
            (
                a,
                b
            ) => a ++ b,
            tripletFields = TripletFields.Dst
        )
    }

    def backpropGatherScatterNormalSum(): GradsType = {

        graph.aggregateMessages[GradType](
            triplet => { // Map Function
                if (triplet.dstAttr.messageMap != null) {
                    triplet.sendToSrc(
                        triplet.dstAttr.messageMap(triplet.srcId)
                    )
                }
            },
            (
                a,
                b
            ) => (a, b).zipped.map(_ + _),
            tripletFields = TripletFields.Dst
        )
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
    // TODO: merge all these functions
    def backpropGatherScatterReverseSingleSum() = {
        def mapFunc(
            triplet: EdgeContext[VertexProperty, EdgeProperty, GradType]
        ): Unit = {
            triplet.sendToDst(triplet.srcAttr.grad.map(_ * triplet.attr.weight))
        }

        //
        def reduceFunc(
            a: GradType,
            b: GradType
        ) = {
            (a, b).zipped.map(_ + _)
        }
        val g = reverseGraph
        val fields = TripletFields.Src

        this._backpropGatherScatter[GradType](g, mapFunc, reduceFunc, fields)
    }

    def backpropGatherScatterMapSum(): GradsType = {
        // Aggregate message way
        // val revereGraph = graph.reverse.partitionBy(PARTITION_STRATEGY)

        reverseGraph.aggregateMessages[GradType](
            triplet => { // Map Function
                // Send message to destination vertex containing
                if ((triplet.srcAttr.messageMap != null)) {
                    triplet.sendToDst(
                        triplet.srcAttr.messageMap(triplet.dstId)
                    )
                }
            },
            // Add counter and age
            (
                a,
                b
            ) => (a, b).zipped.map(_ + _),
            tripletFields = TripletFields.Src
        )
    }

    def backpropGatherScatterMap(): MessageType = {
        // Aggregate message way
        // val revereGraph = graph.reverse.partitionBy(PARTITION_STRATEGY)

        reverseGraph.aggregateMessages[GatherArrayType](
            triplet => { // Map Function
                // Send message to destination vertex containing
                if ((triplet.srcAttr.messageMap != null)) {
                    triplet.sendToDst(
                        Array(
                            (
                                triplet.srcId,
                                triplet.srcAttr.messageMap(triplet.dstId)
                            )
                        )
                    )
                }
            },
            // Add counter and age
            (
                a,
                b
            ) => a ++ b,
            tripletFields = TripletFields.Src
        )
    }

    def gatherScatterCollect(direction: String = "In") = {
        val dir = if (direction == "In") EdgeDirection.In else EdgeDirection.Out
        graph.collectNeighbors(dir)
    }

    def backpropGatherScatterCollect(direction: String = "In") = {
        val dir = if (direction == "In") EdgeDirection.In else EdgeDirection.Out
        graph.reverse.collectNeighbors(dir)
    }

    

    def updateEmbs(updatedEmbs: EmbsType) = {
        // always on the forward graph
        this.graph = this.graph
            .joinVertices(updatedEmbs)((id, vprop, updatedEmb) => {
                // vprop.messageArray = message
                new VertexProperty(num = vprop.num, emb = updatedEmb)
            })
    }

    def updateGrads(gradsMap: Either[MessageType, GradsType]) = {
        updateGraph(gradsMap)
    }

    def getGraphToUpdate() = {
        if (pargs('noReverseGraph) == 1) {
            graph
        } else {
            reverseGraph
        }
    }

    def updateInternalGraph(tmpGraph: GraphType) = {
        if (pargs('noReverseGraph) == 1) {
            this.graph = tmpGraph
        } else {
            this.reverseGraph = tmpGraph
        }
    }

    def updateGraph(messages: Either[MessageType, GradsType]) = {

        var tmpGraph = this.getGraphToUpdate()
        tmpGraph = messages match {
            case Left(x) =>
                tmpGraph
                    .joinVertices(x)((id, vprop, message) => {
                        // vprop.messageArray = message
                        new VertexProperty(
                            num = vprop.num,
                            messageMap = message.toMap
                        )
                    })
            case Right(x) =>
                tmpGraph
                    .joinVertices(x)((id, vprop, message) => {
                        // vprop.messageArray = message
                        new VertexProperty(num = vprop.num, grad = message)
                    })
        }

        // tmpGraph = tmpGraph
        //     .joinVertices(messages)((id, vprop, message) => {
        //         // vprop.messageArray = message
        //         new VertexProperty(num = vprop.num, messageMap = message.toMap)
        //     })

        // if (pargs('isCache) == 1) {
        //     cacheGraph(tmpGraph)
        // } else {
        //     persistGraph(tmpGraph)
        // }
        this.updateInternalGraph(tmpGraph)
    }

    // def updateGraphSingleGrad(messages: GradsType) = {
    //     var tmpGraph = this.getGraphToUpdate()
    //     tmpGraph = tmpGraph
    //         .joinVertices(messages)((id, vprop, message) => {
    //             // vprop.messageArray = message
    //             new VertexProperty(num = vprop.num, grad = message)
    //         })
    //     this.updateInternalGraph(tmpGraph)
    // }

    

    def pipePyT(
        messages: MessageType,
        distScriptName: String,
        printRDDElement: ((VertexId, GatherArrayType), String => Unit) => Unit
    ): RDD[String] = {
        val updatedEmbsString = messages
            .pipe(
                this.tokenize(distScriptName),
                printRDDElement = printRDDElement
            )
        if (pargs('debug) > 1) {
            updatedEmbsString.foreach(System.out.println(_: String))
        }
        return updatedEmbsString
    }

}

