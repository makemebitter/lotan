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

import breeze.linalg.DenseVector
import com.google.common.io.ByteArrayDataOutput
import com.google.common.io.LittleEndianDataOutputStream
import com.google.common.io.ByteStreams
import java.nio.ByteBuffer
import org.zeromq.ZMQ
import org.zeromq.ZContext
import scala.collection.mutable.ArrayBuffer
import java.util.StringTokenizer
import scala.collection.JavaConverters._
import java.util.concurrent.atomic.AtomicReference
import scala.io.Source
import scala.io.Codec
object PlayGround {
    def main(args: Array[String]): Unit = {
        val idx = 0
        // val sizes = Array[Long](1, 3, 1, 3, 1, 3)
        val ids = Array[Long](32331, 12, 1222)
        val arr0 = DenseVector(Array(1.44f, 2.33f, 2.455f))
        val arr1 = DenseVector(Array(2.44f, 4.33f, 6.455f))
        val arr2 = DenseVector(Array(4.44f, 8.33f, 10.455f))
        val vectors = Array[DenseVector[Float]](arr0, arr1, arr2)
        val sizesStream = ByteStreams.newDataOutput()
        val vecsStream = ByteStreams.newDataOutput()
        val idsStream = ByteStreams.newDataOutput()


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
                floats(i) =
                    ByteBuffer.wrap(byteArray, i * times, times).getFloat()
            }
            return floats
        }
        def toLongArray(byteArray: Array[Byte]): Array[Long] = {
            val times = java.lang.Long.SIZE / java.lang.Byte.SIZE
            val longs = Array.fill(byteArray.length / times)(0L)
            for (i <- longs.indices) {
                longs(i) =
                    ByteBuffer.wrap(byteArray, i * times, times).getLong()
            }
            return longs
        }

        for ((id, vec) <- (ids, vectors).zipped) {
            sizesStream.writeLong(1)
            sizesStream.writeLong(vec.length)
            idsStream.writeLong(id)
            vec.valuesIterator.foreach(x => vecsStream.writeFloat(x))
        }
        val sizesByteArray = sizesStream.toByteArray
        val vecsByteArray = vecsStream.toByteArray
        val idsByteArray = idsStream.toByteArray
        

        // val sizesBuffer = ByteBuffer.wrap(sizesByteArray)
        // val vecsBuffer = ByteBuffer.wrap(vecsByteArray)
        // val idsBuffer = ByteBuffer.wrap(idsByteArray)
        

        val command = tokenize("./pipe.py --io_type byte")
        val pb = new ProcessBuilder(command.asJava)
        val currentEnvVars = pb.environment()
        // envVars.foreach { case (variable, value) => currentEnvVars.put(variable, value) }
        val proc = pb.start()

        // Start a thread to print the process's stderr to ours
        
        val encoding = Codec.defaultCharsetCodec.name
        val childThreadException = new AtomicReference[Throwable](null)
        val stderrReaderThread = new Thread(s"$idx $command") {
            override def run(): Unit = {
                val err = proc.getErrorStream
                try {
                    for (
                        line <- Source.fromInputStream(err)(encoding).getLines
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

        

        // Messenger

        // socket
        val zmqContext = new ZContext()
        val socket = zmqContext.createSocket(ZMQ.REQ)
        socket.connect(f"ipc:///mnt/ssd/tmp/messenger_frontend_$idx")
        socket.send(sizesByteArray, ZMQ.SNDMORE)
        socket.send(idsByteArray, ZMQ.SNDMORE)
        socket.send(vecsByteArray, 0)

        // terminate
        socket.send(Array[Byte](), 0)

        // three parts
        val newSizes = toLongArray(socket.recv(0))
        val newIds = toLongArray(socket.recv(0))
        val newVecs = toFloatArray(socket.recv(0))

        //

        // received
        

        var vecPos = 0
        val iter: Iterator[(Long, DenseVector[Float])] =
            for (
                (id, i) <- newIds.zipWithIndex.iterator
            ) yield {
                
                val length = newSizes(i * 2 + 1)
                println(length)
                val vec = newVecs.slice(vecPos, vecPos + length.toInt)
                vecPos = vecPos + length.toInt
                (id, DenseVector[Float](vec))
            }
        // iter.iterator

    }
}
