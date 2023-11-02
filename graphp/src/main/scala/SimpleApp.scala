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

/* SimpleApp.scala */
package org.apache.spark.graphx.lotan.main
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import graphgenerators.GraphGenerators
import constants.Constants
import constants.Types
// import constants.VertexProperty
import constants.EdgeProperty
import scala.collection.mutable.{Map => mMap}
// import shapeless.HMap
import org.apache.ivy.Main
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import upickle.default._
import scala.reflect.ClassTag
import scala.math.sqrt
import org.apache.hadoop.shaded.org.checkerframework.checker.units.qual.g
import lotan.GraphProp
import lotan.GraphPropMKII
import lotan.OneOf
import lotan.First
import lotan.Second
import lotan.Third
import lotan.Fourth
import constants.TypesMKII
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import com.google.common.io.ByteStreams
import scala.collection.mutable.ArrayBuffer
import java.util.StringTokenizer
import java.nio.ByteBuffer
import scala.collection.JavaConverters._
import java.util.concurrent.atomic.AtomicReference
import scala.io.Source
import scala.io.Codec
import org.zeromq.ZMQ
import org.zeromq.ZContext

abstract class IOBase[T] extends TypesMKII {
    val CHAR_MOD = "%e"

    def globalReadEmbds(x: String): EmbsRDDType = {
        val splitted = x.split(",")
        val idx = splitted(0).toFloat.toLong
        val (_, arr) = splitted.splitAt(1)
        (idx, DenseVector(arr.map(_.toFloat)))
    }
    def globalWriteEmbds(x: EmbsRDDType): String = {
        val b = StringBuilder.newBuilder

        b append x._1.toString
        for (y <- x._2.valuesIterator) {
            b append ","
            // b append f"$x%.4f"
            // b append y.toString
            b append CHAR_MOD.format(y)
        }
        b.toString
    }

    def globalWrite(x: (VertexId, T)): String
    def printRDDElementGrad(
        x: (VertexId, T),
        f: String => Unit
    ) = {
        val jsonString = globalWrite(x)
        f(jsonString)
    }
}

// IOObj
// object IOObj extends IOBase {
//     lazy implicit val fooReadWrite: ReadWriter[EmbsRDDType] =
//         readwriter[(Long, Array[Float])].bimap[EmbsRDDType](
//             x => (x._1, x._2.toArray),
//             y => (y._1, DenseVector(y._2))
//         )

//     override def globalRead(x: String) = {
//         read[EmbsRDDType](x)
//     }

//     override def globalWrite(x: EmbsRDDType):String = {
//         write((x._1, x._2.toArray))
//     }
// }

// raw string
object IOObj extends IOBase[DenseVector[Float]] {

    def globalRead = globalReadEmbds(_)

    override def globalWrite(x: EmbsRDDType): String = {
        globalWriteEmbds(x)
    }
}

object IOObjPlain extends IOBase[Array[(VertexId, DenseVector[Float])]] {
    // Gather array out single vector in
    override def globalWrite(x: GatherArrayRDDType): String = {
        val b = StringBuilder.newBuilder

        // first number is vertex id
        b append x._1.toString
        // second number is length of gatherarray
        b append ","
        b append x._2.length.toString
        for (gatherArrayElement <- x._2) {
            // first is id
            b append ","
            b append gatherArrayElement._1.toString
            // then the emb
            for (y <- gatherArrayElement._2.valuesIterator) {
                b append ","
                b append CHAR_MOD.format(y)
            }
        }
        // for (y <- x._2.valuesIterator) {
        //     b append ","
        //     // b append f"$x%.4f"
        //     // b append y.toString
        //     b append CHAR_MOD.format(y)
        // }
        b.toString
    }
    def printRDDElementGradEmb(
        x: (VertexId, EmbType),
        f: String => Unit
    ) = {
        val jsonString = globalWriteEmbds(x)
        f(jsonString)
    }
    override def printRDDElementGrad(
        x: (VertexId, GatherArrayType),
        f: String => Unit
    ) = {
        val jsonString = globalWrite(x)
        f(jsonString)
    }

    def globalRead(x: String): MapOfGradsRDDType = {
        val splitted = x.split(",")
        val idx = splitted(0).toFloat.toLong
        val totalLen = splitted(1).toFloat.toInt
        val eachLen = (splitted.length - 2) / totalLen
        val (_, payload) = splitted.splitAt(2)
        val res = payload
            .grouped(eachLen)
            .map(arr =>
                (
                    arr(0).toFloat.toLong,
                    DenseVector(arr.splitAt(1)._2.map(_.toFloat))
                )
            )
            .toArray

        (idx, res)
        // (idx, DenseVector(arr.map(_.toFloat)))
    }
}

// object IOObjPlainBackward extends IOBase[DenseVector[Float]] {

//     def globalRead(x: String): MapOfGradsRDDType = {
//         val splitted = x.split(",")
//         val idx = splitted(0).toFloat.toLong
//         val totalLen = splitted(1).toFloat.toInt
//         val eachLen = (splitted.length - 2) / totalLen
//         val (_, payload) = splitted.splitAt(2)
//         val res = payload
//             .grouped(eachLen)
//             .map(arr =>
//                 (
//                     arr(0).toFloat.toLong,
//                     DenseVector(arr.splitAt(1)._2.map(_.toFloat))
//                 )
//             )
//             .toArray

//         (idx, res)
//         // (idx, DenseVector(arr.map(_.toFloat)))
//     }

//     override def globalWrite(x: EmbsRDDType): String = {
//         globalWriteEmbds(x)
//     }
// }

// Sparse
object IOObjSparse extends IOBase[SparseVector[Float]] {

    def globalRead(x: String): (VertexId, SparseVector[Float]) = {
        val splitted = x.split(";", -1)
        val idx = splitted(0).toFloat.toLong
        val length = splitted(1).toFloat.toInt
        if (splitted(2) == "") {
            // return empty
            return (
                idx,
                new SparseVector[Float](
                    Array[Int](),
                    Array[Float](),
                    length
                )
            )
        }
        val index = splitted(2).split(",")
        val values = splitted(3).split(",")
        (
            idx,
            new SparseVector[Float](
                index.map(_.toFloat.toInt),
                values.map(_.toFloat.toFloat),
                length
            )
        )
    }
    def globalWrite(x: (VertexId, SparseVector[Float])): String = {
        val b = StringBuilder.newBuilder

        b append x._1.toString
        b append f";${x._2.length};"
        var first = true
        for (y <- x._2.keysIterator) {
            if (first) {
                b append y.toString
                first = false
            } else {
                b append ","
                // b append f"$x%.4f"
                b append y.toString
                // b append CHAR_MOD.format(y)
            }
        }
        b append ";"
        first = true
        for (y <- x._2.valuesIterator) {
            if (first) {
                b append CHAR_MOD.format(y)
                first = false
            } else {
                b append ","
                // b append f"$x%.4f"
                b append CHAR_MOD.format(y)
                // b append CHAR_MOD.format(y)
            }
        }
        b.toString
    }
}

object ByteIOObj extends TypesMKII {
    def tokenize(command: String): Seq[String] = {
        val buf = new ArrayBuffer[String]
        val tok = new StringTokenizer(command)
        while (tok.hasMoreElements) {
            buf += tok.nextToken()
        }
        buf.toSeq
    }
    def toFloatArray(byteArray: Array[Byte]): Array[Float] = {
        val times = java.lang.Float.SIZE / java.lang.Byte.SIZE
        val floats = Array.fill(byteArray.length / times)(0.0f)
        for (i <- floats.indices) {
            floats(i) = ByteBuffer.wrap(byteArray, i * times, times).getFloat()
        }
        return floats
    }
    def toLongArray(byteArray: Array[Byte]): Array[Long] = {
        val times = java.lang.Long.SIZE / java.lang.Byte.SIZE
        val longs = Array.fill(byteArray.length / times)(0L)
        for (i <- longs.indices) {
            longs(i) = ByteBuffer.wrap(byteArray, i * times, times).getLong()
        }
        return longs
    }

    def mapParFn(
        idx: Int,
        iter: Either[Iterator[(VertexId, EmbType)], Iterator[
            (VertexId, EmbSparseType)
        ]],
        epoch: Int,
        baseName: String,
        direction: String,
        layerIDX: Int
    ) = {
        val context = org.apache.spark.TaskContext.get()
        val taskID = context.partitionId()
        // init
        var sizesStream = ByteStreams.newDataOutput()
        var vecsStream = ByteStreams.newDataOutput()
        var idsStream = ByteStreams.newDataOutput()
        var keysStream = ByteStreams.newDataOutput()
        iter match {
            case Left(x) => {
                for ((id, vec) <- x) {
                    sizesStream.writeLong(1)
                    sizesStream.writeLong(vec.length)
                    idsStream.writeLong(id)
                    vec.valuesIterator.foreach(x => vecsStream.writeFloat(x))
                }
            }
            case Right(x) => {
                for ((id, vec) <- x) {
                    sizesStream.writeLong(1)
                    sizesStream.writeLong(vec.length)
                    sizesStream.writeLong(vec.activeSize)
                    idsStream.writeLong(id)
                    vec.activeValuesIterator.foreach(x =>
                        vecsStream.writeFloat(x)
                    )
                    vec.activeKeysIterator.foreach(x => keysStream.writeLong(x))
                }
            }
        }

        var sizesByteArray = sizesStream.toByteArray
        var vecsByteArray = vecsStream.toByteArray
        var idsByteArray = idsStream.toByteArray
        var keysByteArray = keysStream.toByteArray

        // ------------------------- Messenger ------------------------
        val mesID = f"${idx}_${epoch}_${direction}_${layerIDX}"
        val command = tokenize(
            baseName + f" --messenger_idx $mesID"
        )
        val pb = new ProcessBuilder(command.asJava)
        val proc = pb.start()
        val encoding = Codec.defaultCharsetCodec.name
        val childThreadException = new AtomicReference[Throwable](null)
        val stderrReaderThread = new Thread(s"$idx $command") {
            override def run(): Unit = {
                val err = proc.getErrorStream
                try {
                    for (
                        line <- Source
                            .fromInputStream(err)(encoding)
                            .getLines
                    ) {
                        // scalastyle:off println
                        System.err.println(line)
                        // scalastyle:on println
                    }
                } catch {
                    case t: Throwable => childThreadException.set(t)
                } finally {
                    err.close()
                }
            }
        }
        stderrReaderThread.start()
        // ------------------------------------------------------------
        // System.err.println(f"Messenger setup, Task ID: $taskID")
        // ------------------------- Socket ---------------------------
        val zmqContext = new ZContext()
        val socket = zmqContext.createSocket(ZMQ.REQ)
        socket.connect(f"ipc://@messenger_frontend_$mesID")
        // System.err.println(f"SocketSetup, Task ID: $taskID")

        // ------------------------- Iter -----------------------------

        socket.send(sizesByteArray, ZMQ.SNDMORE)
        sizesByteArray = null
        sizesStream = null
        socket.send(idsByteArray, ZMQ.SNDMORE)
        idsByteArray = null
        idsStream = null
        val vecsMore = iter match {
            case Left(x)  => 0
            case Right(x) => ZMQ.SNDMORE
        }
        socket.send(vecsByteArray, vecsMore)
        vecsByteArray = null
        vecsStream = null

        iter match {
            case Left(x) => {}
            case Right(x) => {
                socket.send(keysByteArray, 0)
                keysByteArray = null
                keysStream = null
            }
        }

        // System.err.println(f"SendFinished, Task ID: $taskID")

        var iterRet: Either[Iterator[(VertexId, EmbType)], Iterator[
            (VertexId, EmbSparseType)
        ]] = null

        if (direction == "final") {
            // delivery acck
            val dummy = socket.recv(0)

            // wait for finish
            socket.send(Array[Byte](), 0)
            // only one reply
            val finish = socket.recv(0)
            iterRet = iter match {
                case Left(x) =>
                    Left(
                        Array[(Long, EmbType)](
                            (12345, DenseVector[Float](1.0f, 2.0f))
                        ).iterator
                    )
                case Right(x) =>
                    Right(
                        Array[(Long, EmbSparseType)](
                            (12345, SparseVector[Float](1.0f, 2.0f))
                        ).iterator
                    )
            }
            assert(socket.hasReceiveMore == false)
        } else {
            // three parts
            val newSizes = toLongArray(socket.recv(0))
            val newIds = toLongArray(socket.recv(0))
            val newVecs = toFloatArray(socket.recv(0))

            iter match {
                case Left(_) => {
                    var vecPos = 0
                    iterRet = Left(
                        for ((id, i) <- newIds.zipWithIndex.iterator) yield {

                            val length = newSizes(i * 2 + 1)
                            val vec =
                                newVecs.slice(vecPos, vecPos + length.toInt)
                            vecPos = vecPos + length.toInt
                            // println(length, id, vec)
                            (id, DenseVector[Float](vec))

                        }
                    )

                }
                case Right(_) => {
                    val newKeys = toLongArray(socket.recv(0))
                    var vecPos = 0
                    iterRet = Right(
                        for ((id, i) <- newIds.zipWithIndex.iterator) yield {
                            val sparseArrLength = newSizes(i * 3 + 1)
                            val valuesLength = newSizes(i * 3 + 2)

                            val vec =
                                newVecs.slice(
                                    vecPos,
                                    vecPos + valuesLength.toInt
                                )
                            val keys =
                                newKeys.slice(
                                    vecPos,
                                    vecPos + valuesLength.toInt
                                )
                            vecPos = vecPos + valuesLength.toInt
                            // println(length, id, vec)
                            (
                                id,
                                new SparseVector[Float](
                                    keys.map(_.toInt),
                                    vec,
                                    sparseArrLength.toInt
                                )
                            )

                        }
                    )
                }
            }

        }
        // System.err.println(f"ReceiveFinished, Task ID: $taskID")
        def propagateChildException(): Unit = {
            val t = childThreadException.get()
            if (t != null) {
                val commandRan = command.mkString(" ")
                System.err.println(
                    s"Caught exception while running pipe() operator. Command ran: $commandRan. " +
                        s"Exception: ${t.getMessage}"
                )
                proc.destroy()
                throw t
            }
        }
        // terminate
        socket.send(Array[Byte](), 0)
        val exitStatus = proc.waitFor()
        // System.err.println(f"ProcGracefulExit, Task ID: $taskID")
        if (exitStatus != 0) {
            throw new IllegalStateException(
                s"Subprocess exited with status $exitStatus. " +
                    s"Command ran: " + command.mkString(" ")
            )
        }
        propagateChildException()

        // System.err.println(f"End, Task ID: $taskID")
        // clean up
        context.addTaskCompletionListener[Unit] { _ =>
            if (proc.isAlive) {
                proc.destroy()
            }
            if (stderrReaderThread.isAlive) {
                stderrReaderThread.interrupt()
            }
        }
        // socket.close()
        zmqContext.destroy()

        iterRet

    }

    def pipeCast(
        rdd: Either[EmbsType, EmbsSparseType],
        direction: String,
        layerIDX: Int,
        epoch: Int,
        sparse: Boolean = false
    ): Either[RDD[(VertexId, EmbType)], RDD[(VertexId, EmbSparseType)]] = {
        var baseName =
            "./pipe.py --io_type byte --ipc_type shm --worker_type prebatch_worker"

        rdd match {
            case Right(x) => baseName += " --sparse"
            case _ => {}
        }
        if ((direction == "forward") || (direction == "forbackward")) {
            baseName += " --messenger_single_vector"
        }
        if (direction == "backward") {
            baseName += " --messenger_backprop --messenger_single_vector"
        } else if (direction == "final") {
            baseName += " --messenger_final_backprop"
        }
        val updatedRDD = rdd match {
            case Left(x) =>
                Left(
                    x.mapPartitionsWithIndex(
                        (idx, iter) => {
                            mapParFn(
                                idx,
                                Left(iter): Either[Iterator[
                                    (VertexId, EmbType)
                                ], Iterator[(VertexId, EmbSparseType)]],
                                epoch,
                                baseName,
                                direction,
                                layerIDX
                            ) match {
                                case Left(x) => x
                                case Right(_) => throw new Exception("Not supposed to get here")
                            }

                        },
                        preservesPartitioning = true
                    )
                )
            case Right(x) =>
                Right(
                    x.mapPartitionsWithIndex(
                        (idx, iter) => {
                            mapParFn(
                                idx,
                                Right(iter): Either[Iterator[
                                    (VertexId, EmbType)
                                ], Iterator[(VertexId, EmbSparseType)]],
                                epoch,
                                baseName,
                                direction,
                                layerIDX
                            ) match {
                                case Right(x) => x
                                case Left(_) => throw new Exception("Not supposed to get here")
                            }

                        },
                        preservesPartitioning = true
                    )
                )
        }

        updatedRDD
    }

    def pipeCastWrapper(
        rdd: Either[EmbsType, EmbsSparseType],
        direction: String,
        layerIDX: Int,
        epoch: Int
    ): Either[VertexRDD[EmbType], VertexRDD[EmbSparseType]] = {

        val updatedRDD = this.pipeCast(rdd, direction, layerIDX, epoch)

        val ret = updatedRDD match {
            case Left(x)  => Left(VertexRDD(x))
            case Right(x) => Right(VertexRDD(x))
        }

        ret
    }

}

class ExpRunner(
    var pargs: mMap[Symbol, Int]
) extends Constants
    with MainBase
    with TypesMKII {
    // val graphProp = new GraphProp(pargs)
    val graphProp = new GraphPropMKII(pargs)
    val DTFORMATTER = DateTimeFormatter.ofPattern(DATE_FORMAT)
    this.graphProp.createSpark()
    // val ReadVertexBr = this.graphProp.sc.broadcast(ReadVertex)

    // val ReadVertex = if (lotan.global_args.io_type == "raw_string") {
    //     IORawString
    // } else {
    //     IOJSON
    // }
    // var distScriptNameForward: String = null
    // var distScriptNameForwardSparse: String = null
    // var distScriptNameBackward: String = null
    // var distScriptNameBackwardSparse: String = null
    // var distScriptNameForbackward: String = null
    val miniBatchSize = pargs('miniBatchSize)

    var baseCMD =
        s"./pipe.py --worker_type prebatch_worker --mini_batch_size $miniBatchSize"

    baseCMD = baseCMD + " --io_type"
    baseCMD = if (pargs('ioType) == 0) {
        baseCMD + " raw_string"
    } else {
        baseCMD + " byte"
    }
    baseCMD = baseCMD + " --ipc_type"
    baseCMD = if (pargs('ipcType) == 0) {
        baseCMD + " socket"
    } else {
        baseCMD + " shm"
    }
    var distScriptName = baseCMD
    val distScriptNameFinal = distScriptName + " --messenger_final_backprop"
    val distScriptNameFinalSparse = distScriptNameFinal + " --sparse"
    val distScriptNameForward = if (pargs('aggPushDown) == 0) {
        distScriptName + " --agg_pushdown False"
    } else {
        distScriptName + " --agg_pushdown True --messenger_single_vector"
    }
    val distScriptNameForbackward = if (pargs('aggPushDown) == 0) {
        distScriptNameForward + " --messenger_plain_forbackward"
    } else {
        this.distScriptNameForward
    }
    val distScriptNameForwardSparse =
        distScriptNameForward + " --sparse"
    val distScriptNameForwardSparseFirst =
        distScriptNameForward + " --sparse --first_sparse_pipe"
    val distScriptNameBackward =
        distScriptNameForward + " --messenger_backprop"
    val distScriptNameBackwardSparse =
        distScriptNameBackward + " --sparse"

    val distScriptNameForbackwardSparse =
        distScriptNameForbackward + " --sparse"

    def train(): Unit = {
        graphProp.sc.addFile(LOTAN_NFS_ROOT + "gsys/pipe.py")

        if (pargs('run) == 1 && pargs('fillFeatures) == 1) {

            println("Printing information about the graph and 3 sample vertices and edges")
            graphProp.printGraph(graphProp.graph)

            println("Printing information about the **reverse** graph and 3 sample vertices and edges")
            graphProp.printGraph(graphProp.reverseGraph)
            graphProp.gc(true)
            var messages: OneOf[
                MessageType,
                EmbsType,
                MessageSparseType,
                EmbsSparseType
            ] = null
            for (epoch <- 0 until pargs('numEpochs)) {
                println(s"EPOCH: $epoch")
                for (layer <- 0 until pargs('numLayers) - 1) {
                    val layerIDX = layer
                    // AXW = (AX)W, sum aggregation first in graphx
                    // messages dimension: N x D, #nodes x feature dimension
                    val pipeScript = if (layer == 0) {
                        if (this.pargs('sparse) == 1) {
                            distScriptNameForwardSparseFirst
                        } else {
                            distScriptNameForward
                        }

                    } else {
                        if (this.pargs('sparse) == 1) {
                            distScriptNameForwardSparse
                        } else {
                            distScriptNameForward
                        }

                    }
                    val messages = this.forwardOverall(epoch, layerIDX)
                    if(DEBUG){
                        messages match {
                            case First(x)  => x.take(10).foreach(println)
                            case Second(x) => x.take(10).foreach(println)
                            case Third(x)  => x.take(10).foreach(println)
                            case Fourth(x) => x.take(10).foreach(println)
                        }
                    }
                    // gc
                    graphProp.resetCache(true)

                    this.pipeCastJoinForwardOverall(
                        messages,
                        epoch,
                        layerIDX,
                        pipeScript
                    )
                }

                // last layer propogation + back prop
                // AH
                var layerIDX = pargs('numLayers) - 1
                var messages =
                    this.forwardOverall(epoch, layerIDX = layerIDX)

                // gc graph
                graphProp.resetCache(true)

                val backwardOutputFormat =
                    if (pargs('aggPushDown) == 1) {
                        "single"
                    } else {
                        "map"
                    }

                this.pipeCastJoinForbackwardOverall(
                    messages,
                    epoch,
                    layerIDX,
                    direction = "forbackward",
                    outFormat = backwardOutputFormat
                )

                for (layer <- 0 until pargs('numLayers) - 2) {
                    // last to last - 1 layer, get grads

                    // graphProp.updateGrads(gradsSingle)

                    //
                    val layerIDX = pargs('numLayers) - 2 - layer
                    val messages = this.backwardAggPushDown(epoch, layerIDX)
                    // gc graph
                    graphProp.resetCache(true)

                    this.pipeCastJoinForbackwardOverall(
                        messages,
                        epoch,
                        layerIDX,
                        direction = "backward",
                        outFormat = backwardOutputFormat
                    )

                }

                // last layer
                layerIDX = 0

                messages = this.backwardAggPushDown(epoch, layerIDX)
                // gc graph
                graphProp.resetCache(true)
                // val grads = graphProp.backpropGatherScatterReverseSingleSum()

                // grads
                //     .map { case record => write(record) }
                //     .saveAsTextFile(
                //         s"/mnt/nfs/ssd/lognormal/lognormal_grads_second_layer_$epoch.txt"
                //     )

                // final backprop in py
                // backward uses the vanilla messenger
                this.graphProp.spark.time {
                    if (pargs('ioType) == 0) {

                        val finalTags = messages match {
                            case Second(m) => {
                                this.pipe(
                                    epoch,
                                    layerIDX,
                                    m,
                                    distScriptNameFinal,
                                    IOObj.printRDDElementGrad
                                )
                            }
                            case Fourth(m) => {
                                this.pipe(
                                    epoch,
                                    layerIDX,
                                    m,
                                    distScriptNameFinalSparse,
                                    IOObjSparse.printRDDElementGrad
                                )
                            }
                            case _ => null
                        }
                        // val finalTags = grads
                        //     .pipe(
                        //         graphProp.tokenize(distScriptNameFinal),
                        //         printRDDElement = printRDDElementGrad
                        //     )
                        finalTags.foreach(System.out.println(_: String))
                    } else if (pargs('ioType) == 1) {
                        val finalTags = messages match {
                            case Second(m) =>
                                ByteIOObj.pipeCast(
                                    Left(m),
                                    "final",
                                    layerIDX,
                                    epoch
                                )
                            case Fourth(m) =>
                                ByteIOObj.pipeCast(
                                    Right(m),
                                    "final",
                                    layerIDX,
                                    epoch
                                )

                            // gradsmap not implemented
                            case _ =>
                                null
                        }
                        // trigger
                        finalTags match {
                            case Left(x)  => x.count()
                            case Right(x) => x.count()
                        }
                    }

                    // get rid of messages
                    messages match {
                        case First(x)  => this.graphProp.uncacheRDD(x)
                        case Second(x) => this.graphProp.uncacheRDD(x)
                        case Third(x)  => this.graphProp.uncacheRDD(x)
                        case Fourth(x) => this.graphProp.uncacheRDD(x)
                    }
                    this.logDrillDown(epoch, layerIDX, "pipeCast")
                }
                graphProp.graph = graphProp.originalGraph
                graphProp.reverseGraph = graphProp.originalReverseGraph
                graphProp.gc()

            }

        }
        // graphProp.spark.stop()
        this.graphProp.sendTerm()

    }

    def run(): Unit = {

        if (pargs('dataset) == 100) {
            graphProp.genRandomGraph()
        } else {
            graphProp.loadGraph()
        }

        if (pargs('hardPartition) != 1) {
            this.train()
        } else {
            graphProp.hardPartition()
        }

    }

    def printTrigger[T](vRDD: VertexRDD[Array[T]]): Unit = {
        val rsamples = vRDD.take(10)
        for ((id, arr) <- rsamples) {
            val arr_str = arr.mkString(", ")
            println(s"ID: $id, Content: $arr_str")
        }
    }

    def forwardOverall(epoch: Int, layerIDX: Int): OneOf[
        MessageType,
        EmbsType,
        MessageSparseType,
        EmbsSparseType
    ] = {
        var res: OneOf[
            MessageType,
            EmbsType,
            MessageSparseType,
            EmbsSparseType
        ] = null
        this.graphProp.spark.time {

            res = if (pargs('aggPushDown) == 0) {
                First(this.forward(epoch, layerIDX))
            } else {
                this.forwardAggPushDown(epoch, layerIDX) match {
                    case Left(x)  => Second(x)
                    case Right(x) => Fourth(x)
                }

            }
            this.logDrillDown(epoch, layerIDX, "gatherScatterSumAgg")
        }
        return res
    }

    def forward(epoch: Int, layerIDX: Int): MessageType = {
        // ----------------------- array method -----------------
        // forward prop
        val messages = graphProp.gatherScatter(layerIDX)
        if (pargs('drillDown) == 1) {
            messages.cache()
            messages.count()
        }
        // if (pargs('verbose) > 0) {
        //     val messages_count = messages.count()
        //     println(s"messages: #vertices: $messages_count")
        //     printTrigger(messages)
        //     // val msamples = messages.take(10)
        //     // for ((id, arr) <- msamples) {
        //     //     val arr_str = arr.mkString(", ")
        //     //     println(s"ID: $id, Content: $arr_str")
        //     // }
        // }
        return messages
        // ------------------------------------------------------
    }

    def forwardAggPushDown(
        epoch: Int,
        layerIDX: Int
    ): Either[EmbsType, EmbsSparseType] = {
        // var messages: Either[EmbsType, EmbsSparseType] = null

        val messages = this.graphProp.gatherScatterSumAgg(layerIDX = layerIDX)

        if (pargs('drillDown) == 1) {
            messages match {
                case Left(x) => {
                    this.graphProp.cacheRDD(x, "gscmessage", "tmp")
                }
                case Right(x) => {
                    this.graphProp.cacheRDD(x, "gscmessage", "tmp")
                }
            }

        }
        return messages
    }

    def backwardAggPushDown(
        epoch: Int,
        layerIDX: Int
    ): OneOf[
        MapOfGradsType,
        GradsType,
        MapOfGradsSparseType,
        GradsSparseType
    ] = {
        var res: OneOf[
            MapOfGradsType,
            GradsType,
            MapOfGradsSparseType,
            GradsSparseType
        ] = null
        var grads: Either[GradsType, GradsSparseType] = null
        this.graphProp.spark.time {

            grads = if (pargs('aggPushDown) == 1) {
                this.graphProp.backpropGatherScatterSingleSum()
            } else {
                Left(this.graphProp.backpropGatherScatter())
            }
            if (pargs('drillDown) == 1) {
                grads match {
                    case Left(x) => {
                        this.graphProp.cacheRDD(x, "gscgrads", "tmp")
                        res = Second(x)
                    }
                    case Right(x) => {
                        this.graphProp.cacheRDD(x, "gscgrads", "tmp")
                        res = Fourth(x)
                    }
                }

            }
            this.logDrillDown(
                epoch,
                layerIDX,
                "backpropGatherScatterReverseSingleSum"
            )
            // println(s"Epoch: $epoch, layerIDX: $layerIDX, gatherScatterSumAgg")
        }
        return res
    }

    def logDrillDown(epoch: Int, layerIDX: Int, method: String) = {
        println(s"Epoch: $epoch, layerIDX: $layerIDX, $method")
    }
    def pipe[T](
        epoch: Int,
        layerIDX: Int,
        messages: RDD[T],
        distScriptNameForward: String,
        printRDDElement: (T, String => Unit) => Unit
    ): RDD[String] = {
        var updatedEmbsString: RDD[String] = null
        this.graphProp.spark.time {
            updatedEmbsString = messages
                .pipe(
                    graphProp.tokenize(distScriptNameForward),
                    printRDDElement = printRDDElement
                )
            if (pargs('drillDown) == 1) {
                updatedEmbsString.cache()
                updatedEmbsString.count()
            }
            this.logDrillDown(epoch, layerIDX, "pipe")
            // println(s"Epoch: $epoch, layerIDX: $layerIDX, pipe")
        }
        // get rid of messages
        this.graphProp.uncacheRDD(messages)
        return updatedEmbsString
    }
    // def cast(
    //     epoch: Int,
    //     layerIDX: Int,
    //     updatedEmbsString: RDD[String]
    // ): EmbsType = {
    //     var updatedEmbs: EmbsType = null
    //     this.graphProp.spark.time {
    //         updatedEmbs = VertexRDD(
    //             updatedEmbsString.map(x => read[EmbsRDDType](x))
    //         )
    //         if (pargs('drillDown) == 1) {
    //             updatedEmbs.cache()
    //             updatedEmbs.count()
    //         }
    //         this.logDrillDown(epoch, layerIDX, "cast")
    //     }
    //     return updatedEmbs
    // }
    def cast(
        epoch: Int,
        layerIDX: Int,
        updatedEmbsString: RDD[String]
    ): Either[EmbsType, EmbsSparseType] = {
        if (this.pargs('sparse) != 1) {
            var updatedEmbs: EmbsType = null

            updatedEmbs = VertexRDD(
                updatedEmbsString.map(x => IOObj.globalRead(x))
            )
            if (pargs('drillDown) == 1) {
                updatedEmbs.cache()
                updatedEmbs.count()
            }

            return Left(updatedEmbs)

        } else {
            var updatedEmbs: EmbsSparseType = null
            updatedEmbs = VertexRDD(
                updatedEmbsString.map(x => IOObjSparse.globalRead(x))
            )
            if (pargs('drillDown) == 1) {
                updatedEmbs.cache()
                updatedEmbs.count()
            }
            return Right(updatedEmbs)
        }
    }

    def castMap(
        epoch: Int,
        layerIDX: Int,
        updatedEmbsString: RDD[String]
    ): Either[MapOfGradsType, MapOfGradsSparseType] = {

        if (this.pargs('sparse) != 1) {
            var updatedEmbs: MapOfGradsType = null
            updatedEmbs = VertexRDD(
                updatedEmbsString.map(x => IOObjPlain.globalRead(x))
            )
            if (pargs('drillDown) == 1) {
                updatedEmbs.cache()
                updatedEmbs.count()
            }

            return Left(updatedEmbs)

        } else {
            return null
        }
    }

    def castOverall(
        epoch: Int,
        layerIDX: Int,
        updatedEmbsString: RDD[String],
        outFormat: String = "single"
    ): OneOf[
        MapOfGradsType,
        GradsType,
        MapOfGradsSparseType,
        GradsSparseType
    ] = {
        var res: OneOf[
            MapOfGradsType,
            GradsType,
            MapOfGradsSparseType,
            GradsSparseType
        ] = null
        this.graphProp.spark.time {
            res = if (outFormat == "map") {
                println("Casting Map")

                val casted = this.castMap(epoch, layerIDX, updatedEmbsString)
                casted match {
                    case Left(x)  => First(x)
                    case Right(x) => Third(x)
                }

            } else {
                println("Casting single")
                val casted = this.cast(epoch, layerIDX, updatedEmbsString)
                casted match {

                    case Left(x)  => Second(x)
                    case Right(x) => Fourth(x)
                }
            }
            this.logDrillDown(epoch, layerIDX, "cast")
        }
        return res
    }

    def pipeCastJoinForwardOverall(
        messages: OneOf[
            MessageType,
            EmbsType,
            MessageSparseType,
            EmbsSparseType
        ],
        epoch: Int,
        layerIDX: Int,
        pipeScript: String
    ) = {
        this.graphProp.spark.time {
            messages match {
                case First(x) =>
                    this.pipeCastJoinForwardPlain(
                        Left(x),
                        epoch,
                        layerIDX,
                        pipeScript
                    )
                case Third(x) =>
                    this.pipeCastJoinForwardPlain(
                        Right(x),
                        epoch,
                        layerIDX,
                        pipeScript
                    )
                case Second(x) =>
                    this.pipeCastJoinForward(
                        Left(x),
                        epoch,
                        layerIDX,
                        pipeScript
                    )
                case Fourth(x) =>
                    this.pipeCastJoinForward(
                        Right(x),
                        epoch,
                        layerIDX,
                        pipeScript
                    )
            }

            this.logDrillDown(epoch, layerIDX, "pipeCastJoin")
        }
    }
    def pipeCastJoinForwardPlain(
        messages: Either[MessageType, MessageSparseType],
        epoch: Int,
        layerIDX: Int,
        pipeScript: String
    ) = {
        // string IO
        var updatedEmbsString: RDD[String] = null
        var updatedEmbs: Either[EmbsType, EmbsSparseType] = null
        if (this.pargs('ioType) == 0) {
            this.graphProp.spark.time {
                updatedEmbsString = messages match {
                    case Left(m) =>
                        this.pipe(
                            epoch,
                            layerIDX,
                            m,
                            pipeScript,
                            IOObjPlain.printRDDElementGrad
                        )
                    // sparse not implemented
                    case Right(m) => null
                }
                updatedEmbs = this.cast(epoch, layerIDX, updatedEmbsString)
                this.logDrillDown(epoch, layerIDX, "pipeCast")

            }
            this.join(epoch, layerIDX, updatedEmbs)

        }
        // byte IO not implemented
        else {}
    }

    def pipeCastJoinForward(
        messages: Either[EmbsType, EmbsSparseType],
        epoch: Int,
        layerIDX: Int,
        pipeScript: String
    ) = {
        var updatedEmbsString: RDD[String] = null
        var updatedEmbs: Either[EmbsType, EmbsSparseType] = null
        if (this.pargs('ioType) == 0) {
            // first layer always dense, but pipe might return sparse
            this.graphProp.spark.time {
                updatedEmbsString = messages match {
                    case Left(m) =>
                        this.pipe(
                            epoch,
                            layerIDX,
                            m,
                            pipeScript,
                            IOObj.printRDDElementGrad
                        )
                    // sparse
                    case Right(m) =>
                        this.pipe(
                            epoch,
                            layerIDX,
                            m,
                            pipeScript,
                            IOObjSparse.printRDDElementGrad
                        )

                }
                updatedEmbs = this.cast(epoch, layerIDX, updatedEmbsString)
                this.logDrillDown(epoch, layerIDX, "pipeCast")
            }
            this.join(epoch, layerIDX, updatedEmbs)
        } else if (this.pargs('ioType) == 1) {
            var updatedEmbs: Either[EmbsType, EmbsSparseType] = null
            this.graphProp.spark.time {
                updatedEmbs = messages match {
                    case Left(m) => {
                        val ret = ByteIOObj.pipeCastWrapper(
                            Left(m),
                            "forward",
                            layerIDX,
                            epoch
                        )
                        ret match {
                            case Left(x) => {
                                this.graphProp.cacheRDD(x, "pipedback", "tmp")
                                Left(x)
                            }
                            case _ => throw new Exception("Not supposed to get here")
                        }

                    }
                    // sparse
                    case Right(m) =>
                        val ret = ByteIOObj.pipeCastWrapper(
                            Right(m),
                            "forward",
                            layerIDX,
                            epoch
                        )
                        ret match {
                            case Right(x) => {
                                this.graphProp.cacheRDD(x, "pipedback", "tmp")
                                Right(x)
                            }
                            case _ => throw new Exception("Not supposed to get here")
                        }
                }
                this.logDrillDown(epoch, layerIDX, "pipeCast")
            }
            // get rid of messages
            messages match {
                case Left(x)  => this.graphProp.uncacheRDD(x)
                case Right(x) => this.graphProp.uncacheRDD(x)
            }

            this.join(epoch, layerIDX, updatedEmbs)

        }

    }

    def pipeCastJoinForbackwardOverall(
        messages: OneOf[
            MessageType,
            EmbsType,
            MessageSparseType,
            EmbsSparseType
        ],
        epoch: Int,
        layerIDX: Int,
        direction: String,
        outFormat: String = "single"
    ) = {
        if (this.pargs('ioType) == 0) {
            // softmax(_ W), labels, validation markup, and backprop
            // each vertex has one grad
            var scriptName: String = null
            scriptName = messages match {
                case aOrb @ (First(_) | Second(_)) =>
                    if (direction == "forward") { this.distScriptNameForward }
                    else if (direction == "forbackward")
                        (this.distScriptNameForbackward)
                    else { this.distScriptNameBackward }
                case aOrb @ (Third(_) | Fourth(_)) =>
                    if (direction == "forward") {
                        this.distScriptNameForwardSparse
                    } else if (direction == "forbackward")
                        (this.distScriptNameForbackwardSparse)
                    else { this.distScriptNameBackwardSparse }
            }

            var grads: OneOf[
                MapOfGradsType,
                GradsType,
                MapOfGradsSparseType,
                GradsSparseType
            ] = null
            var gradsString: RDD[String] = null
            this.graphProp.spark.time {
                gradsString = messages match {
                    case First(m) => {
                        this.pipe(
                            epoch,
                            layerIDX,
                            m,
                            scriptName,
                            IOObjPlain.printRDDElementGrad
                        )

                    }
                    case Second(m) => {
                        this.pipe(
                            epoch,
                            layerIDX,
                            m,
                            scriptName,
                            IOObj.printRDDElementGrad
                        )
                    }
                    case Third(m) => {
                        null
                    }
                    case Fourth(m) => {
                        this.pipe(
                            epoch,
                            layerIDX,
                            m,
                            scriptName,
                            IOObjSparse.printRDDElementGrad
                        )
                    }
                }
                // println("printing grads")
                // gradsString.take(10).foreach(println)
                grads =
                    this.castOverall(epoch, layerIDX, gradsString, outFormat)
                this.logDrillDown(epoch, layerIDX, "pipeCast")
            }
            this.joinGrad(epoch, layerIDX, grads)
        } else if (this.pargs('ioType) == 1) {
            var grads: OneOf[
                MapOfGradsType,
                GradsType,
                MapOfGradsSparseType,
                GradsSparseType
            ] = null

            this.graphProp.spark.time {
                grads = messages match {
                    case Second(m) => {
                        val ret = ByteIOObj.pipeCastWrapper(
                            Left(m),
                            direction,
                            layerIDX,
                            epoch
                        )
                        ret match {
                            case Left(x) => {
                                this.graphProp.cacheRDD(x, "pipedback", "tmp")
                                Second(x)
                            }
                            case Right(_) => throw new Exception("Not supposed to get here")
                        }

                    }
                    // sparse.
                    case Fourth(m) => {
                        val ret = ByteIOObj.pipeCastWrapper(
                            Right(m),
                            direction,
                            layerIDX,
                            epoch
                        )
                        ret match {
                            case Right(x) => {
                                this.graphProp.cacheRDD(x, "pipedback", "tmp")
                                if (DEBUG){
                                    x.take(10).foreach(println)
                                }
                                Fourth(x)
                            }
                            case Left(_) => throw new Exception("Not supposed to get here")
                        }

                    }
                    //  gradsmap not implemented
                    case _ =>
                        null

                }
                // get rid of messages
                messages match {
                    case First(x)  => this.graphProp.uncacheRDD(x)
                    case Second(x) => this.graphProp.uncacheRDD(x)
                    case Third(x)  => this.graphProp.uncacheRDD(x)
                    case Fourth(x) => this.graphProp.uncacheRDD(x)
                }
                this.logDrillDown(epoch, layerIDX, "pipeCast")
            }

            this.joinGrad(epoch, layerIDX, grads)

        }
    }

    // deprecated
    // def pipeCastJoinForbackward(
    //     messages: Either[EmbsType, EmbsSparseType],
    //     epoch: Int,
    //     layerIDX: Int,
    //     direction: String
    // ) = {
    //     if (this.pargs('ioType) == 0) {
    //         // softmax(_ W), labels, validation markup, and backprop
    //         // each vertex has one grad
    //         var scriptName: String = null
    //         if (direction == "forward") {
    //             scriptName = messages match {
    //                 case Left(_)  => this.distScriptNameForward
    //                 case Right(_) => this.distScriptNameForwardSparse
    //             }
    //         } else if (direction == "backward") {
    //             scriptName = messages match {
    //                 case Left(_)  => this.distScriptNameBackward
    //                 case Right(_) => this.distScriptNameBackwardSparse
    //             }
    //         }

    //         var grads: OneOf[
    //             MapOfGradsType,
    //             GradsType,
    //             MapOfGradsSparseType,
    //             GradsSparseType
    //         ] = null
    //         var gradsString: RDD[String] = null

    //         this.graphProp.spark.time {
    //             gradsString = messages match {
    //                 case Left(m) => {
    //                     this.pipe(
    //                         epoch,
    //                         layerIDX,
    //                         m,
    //                         scriptName,
    //                         IOObj.printRDDElementGrad
    //                     )
    //                 }
    //                 case Right(m) => {
    //                     this.pipe(
    //                         epoch,
    //                         layerIDX,
    //                         m,
    //                         scriptName,
    //                         IOObjSparse.printRDDElementGrad
    //                     )
    //                 }
    //             }
    //             grads = this.castOverall(epoch, layerIDX, gradsString)
    //             this.logDrillDown(epoch, layerIDX, "pipeCast")
    //         }
    //         this.joinGrad(epoch, layerIDX, grads)
    //     } else if (this.pargs('ioType) == 1) {
    //         var grads: OneOf[
    //             MapOfGradsType,
    //             GradsType,
    //             MapOfGradsSparseType,
    //             GradsSparseType
    //         ] = null
    //         this.graphProp.spark.time {
    //             grads = messages match {
    //                 case Left(m) =>
    //                     Second(
    //                         ByteIOObj.pipeCastWrapper(
    //                             m,
    //                             direction,
    //                             layerIDX,
    //                             epoch
    //                         )
    //                     )
    //                 // sparse not implemented, gradsmap not implemented
    //                 case Right(m) =>
    //                     null
    //             }
    //             this.logDrillDown(epoch, layerIDX, "pipeCast")
    //         }

    //         this.joinGrad(epoch, layerIDX, grads)

    //     }
    // }

    def join(
        epoch: Int,
        layerIDX: Int,
        updatedEmbs: Either[EmbsType, EmbsSparseType]
    ) = {
        this.graphProp.spark.time {
            this.graphProp.updateEmbs(updatedEmbs)
            if (epoch == 0) {
                // trigger graph compute
                this.graphProp.cacheGraph(this.graphProp.graph, lease = "tmp")
                // this.graphProp.graph.cache()
                this.graphProp.printGraph(this.graphProp.graph)
                this.graphProp.sendFinished()
            }
            if (pargs('drillDown) == 1) {
                // this.graphProp.graph.cache()
                this.graphProp.cacheGraph(this.graphProp.graph, lease = "tmp")
                this.graphProp.graph.vertices.count()
                this.graphProp.graph.edges.count()
            }
            // get rid of updatedEmbs
            updatedEmbs match {
                case Left(x)  => this.graphProp.uncacheRDD(x)
                case Right(x) => this.graphProp.uncacheRDD(x)
            }
            this.logDrillDown(epoch, layerIDX, "join")
        }
    }

    def joinGrad(
        epoch: Int,
        layerIDX: Int,
        gradsSingle: OneOf[
            MapOfGradsType,
            GradsType,
            MapOfGradsSparseType,
            GradsSparseType
        ]
    ) = {
        this.graphProp.spark.time {
            this.graphProp.updateGraph(gradsSingle)
            if (pargs('drillDown) == 1) {
                val g = this.graphProp.getGraphToUpdate()
                this.graphProp.cacheGraph(g, lease = "tmp")
                // g.cache()
                g.vertices.count()
                g.edges.count()
            }
            gradsSingle match {
                case First(x)  => this.graphProp.uncacheRDD(x)
                case Second(x) => this.graphProp.uncacheRDD(x)
                case Third(x)  => this.graphProp.uncacheRDD(x)
                case Fourth(x) => this.graphProp.uncacheRDD(x)
            }
            this.logDrillDown(epoch, layerIDX, "joinGrad")
        }
    }

    // def forwardAndPyT(
    //     distScriptName: String,
    //     printRDDElement: ((VertexId, GatherArrayType), String => Unit) => Unit,
    //     layerIDX: Int
    // ): RDD[String] = {

    //     val messages = forward(layerIDX)
    //     val updatedEmbsString = messages
    //         .pipe(
    //             graphProp.tokenize(distScriptName),
    //             printRDDElement = printRDDElement
    //         )
    //     if (pargs('debug) > 1) {
    //         updatedEmbsString.foreach(System.out.println(_: String))
    //     }
    //     return updatedEmbsString
    // }

    def updateMessages(messages: MessageType) = {}

    // def updateAndBackprop(messages: MessageType): MessageType = {
    //     // update graph
    //     val mes: Either[MapOfGradsType, GradsType] = Left(messages)
    //     graphProp.updateGraph(mes)
    //     if (pargs('verbose) > 0) {
    //         // trigger action
    //         graphProp.printGraph(graphProp.graph)
    //     }
    //     val gradientRDD = graphProp.backpropGatherScatter()
    //     return gradientRDD
    // }
}

// var numVertices: Int = 1e7.toInt,
// var numEParts: Int = 4000,
object SimpleApp extends MainBase {
    def main(args: Array[String]): Unit = {
        // val logFile = "/local/spark/README.md" // Should be some file on your system
        val pargs = parse(args)
        val expRunner = new ExpRunner(pargs)
        expRunner.run()
        println("__SUCCESS__")
    }
}
