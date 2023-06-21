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
    
package constants
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.collection.mutable.{Map => mMap}
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import breeze.linalg.StorageVector
import upickle.default._

abstract class VertexPropertyBase extends Serializable {}
abstract class SingleVec {
    type dataType
    val data: dataType

    def +(): SingleVec
    def *(): SingleVec
}

// class ArraySingleVec(val data: Array[Float]) extends SingleVec{
//     def +(that: ArraySingleVec) = new ArraySingleVec((this.data, that.data).zipped.map(_ + _))

// }

class VertexProperty(
    val num: Long,
    val emb: Array[Float] = null,
    val grad: Array[Float] = null,
    val messageMap: Map[VertexId, Array[Float]] = null,
    val feat: Array[Float] = null,
    val inDegrees: Double = 1.toDouble,
    val outDegrees: Double = 1.toDouble,
    val labeled: Double = 0.toDouble
) extends VertexPropertyBase {
    var message: Array[(VertexId, VertexProperty)] = _
    var messageArray: Array[(VertexId, Array[Float])] = _

    // def updateEmb(newEmb: EmbType): this.type = {
    //     this.emb = newEmb
    //     return this
    // }
    override def toString(): String = {
        var embStr = ""
        var embLen = ""
        var featStr = ""
        var featLen = ""
        val messageStr = if (message != null) {
            message.length.toString()
        } else ""
        val messageArrayStr = if (messageArray != null) {
            messageArray.length.toString()
        } else ""
        val messageMapStr = if (messageMap != null) {
            messageMap.size.toString()
        } else ""
        if (emb != null) {
            // embStr = emb.mkString(", ")
            embLen = emb.length.toString()
        }
        if (feat != null) {
            // featStr = feat.mkString(", ")
            featLen = feat.length.toString()
        }

        s"ID: $num, Degrees(in/out): $inDegrees/$outDegrees, featLen: $featLen, embLen: $embLen, messageLen: $messageStr, messageArrayLen: $messageArrayStr, messageMapLen: $messageMapStr, emb: $embStr"
    }
}

class VertexPropertyMKII(
    val num: Long,
    val emb: DenseVector[Float] = null,
    val grad: DenseVector[Float] = null,
    val messageMap: Map[VertexId, DenseVector[Float]] = null,
    val feat: DenseVector[Float] = null,
    val inDegrees: Double = 1.toDouble,
    val outDegrees: Double = 1.toDouble,
    val vecSparse: SparseVector[Float] = null,
    val labeled: Double = 0.toDouble
) extends VertexPropertyBase {
    var message: Array[(VertexId, VertexPropertyMKII)] = _
    var messageArray: Array[(VertexId, DenseVector[Float])] = _

    // def updateEmb(newEmb: EmbType): this.type = {
    //     this.emb = newEmb
    //     return this
    // }
    override def toString(): String = {
        var embStr = ""
        var embLen = ""
        var featStr = ""
        var featLen = ""
        val messageStr = if (message != null) {
            message.length.toString()
        } else ""
        val messageArrayStr = if (messageArray != null) {
            messageArray.length.toString()
        } else ""
        val messageMapStr = if (messageMap != null) {
            messageMap.size.toString()
        } else ""
        if (emb != null) {
            embStr = emb.toArray.mkString(", ")
            embLen = emb.length.toString()
        }
        if (feat != null) {
            // featStr = feat.mkString(", ")
            featLen = feat.length.toString()
        }

        s"ID: $num, labeled: $labeled, Degrees(in/out): $inDegrees/$outDegrees, featLen: $featLen, embLen: $embLen, messageLen: $messageStr, messageArrayLen: $messageArrayStr, messageMapLen: $messageMapStr, emb: $embStr"
    }
}

class EdgeProperty(
    // val grad: Array[Float] = null,
    val weight: Float = 1.toFloat
) extends Serializable {}

trait Constants {
    val DEBUG = false
    val SEED = 2021
    val EDGE_FILE_PATH = "/mnt/nfs/ssd/lognormal/lognormal_edge.txt"
    val VERTEX_FILE_PATH = "/mnt/nfs/ssd/lognormal/lognormal_vertex.txt"
    val REVERSE_EDGE_FILE_PATH =
        "/mnt/nfs/ssd/lognormal/reverse_lognormal_edge.txt"
    val REVERSE_VERTEX_FILE_PATH =
        "/mnt/nfs/ssd/lognormal/reverse_lognormal_vertex.txt"
    val DATE_FORMAT = "YYYY-mm-dd HH:MM:SS"
    val META = "meta.csv"
    val EDGES = "edges.csv"
    val FEATURES = "features.csv"
    val datasetMap = mMap(
        "products" -> "/mnt/nfs/ssd/products",
        "processed" -> "/mnt/nfs/ssd/processed"
    )
    val datasetBetterMap = mMap(
        0 -> "/mnt/nfs/ssd/processed",
        1 -> "/mnt/nfs/ssd/products",
        2 -> "/mnt/nfs/ssd/arxiv",
        3 -> "hdfs://master:9000/papers100M"
    )
    val hosts = Seq(
        "10.10.1.1",
        "10.10.1.2"
    )
    val master = "spark://10.10.1.1:7077"
    type OptionMap = mMap[Symbol, Int]

}
trait Types {
    type EmbType = Array[Float]
    type EmbsType = VertexRDD[EmbType]
    type EmbsRDDType = (VertexId, EmbType)

    type GatherType = Array[(VertexId, VertexProperty)]
    type GatherArrayType = Array[(VertexId, EmbType)]
    type GatherMapType = Map[VertexId, EmbType]
    type MessageType = VertexRDD[GatherArrayType]

    type GradType = EmbType
    type GradsType = EmbsType
    type GradsRDDType = EmbsRDDType
    type MapOfGradsHomoRDDType = EmbsRDDType
    type MapOfGradsHomoType = EmbsType
    type MapOfGradsType = MessageType
    type MapOfGradsRDDType = (VertexId, GatherArrayType)
    type BackpropGatherType = GradType
    type GraphType = Graph[VertexProperty, EdgeProperty]
    type VertexRDDType = VertexRDD[(VertexId, VertexProperty)]
}

trait TypesMKII {
    // type EmbType = DenseVector[Float]
    type EmbType = DenseVector[Float]
    type EmbSparseType = SparseVector[Float]
    type GraphType = Graph[VertexPropertyMKII, EdgeProperty]
    type VertexRDDType = VertexRDD[(VertexId, VertexPropertyMKII)]

    type EmbsType = VertexRDD[EmbType]
    type EmbsSparseType = VertexRDD[EmbSparseType]
    type EmbsRDDType = (VertexId, EmbType)
    type EmbsSparseRDDType = (VertexId, EmbSparseType)

    type GatherType = Array[(VertexId, VertexPropertyMKII)]
    type GatherArrayType = Array[(VertexId, EmbType)]
    type GatherArraySparseType = Array[(VertexId, EmbSparseType)]
    type GatherMapType = Map[VertexId, EmbType]
    type MessageType = VertexRDD[GatherArrayType]
    type MessageSparseType = VertexRDD[GatherArraySparseType]

    type GradType = EmbType
    type GradSparseType = EmbSparseType
    type GradsType = EmbsType
    type GradsSparseType = EmbsSparseType
    type GradsRDDType = EmbsRDDType
    type MapOfGradsHomoRDDType = EmbsRDDType
    type MapOfGradsHomoType = EmbsType
    type MapOfGradsType = MessageType
    type MapOfGradsSparseType = MessageSparseType
    type MapOfGradsRDDType = (VertexId, GatherArrayType)
    type GatherArrayRDDType = MapOfGradsRDDType
    //
    type BackpropGatherType = GradType

}
// sealed trait Emb
// final case class EmbFloat(x: Array[Float]) extends Emb
// sealed trait Gathered
// final case class GatheredArrayVertexProperty(x: Array[(VertexId, VertexProperty)]) extends Gathered
// final case class GatheredArrayEmb(x: Array[(VertexId, Emb)]) extends Gathered
// final case class GatheredMapEmb(x: Map[VertexId, Emb]) extends Gathered
// sealed trait GatheredRDD
// final case class GatheredArrayRDD(x: VertexRDD[GatheredArrayEmb]) extends GatheredRDD
// final case class GatheredEmbRDD(x: VertexRDD[Emb]) extends GatheredRDD
