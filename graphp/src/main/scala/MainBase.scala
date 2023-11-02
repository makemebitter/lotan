package org.apache.spark.graphx.lotan.main
import constants.Constants
import constants.Types
import scala.collection.mutable.{Map => mMap}

// trait Param(value: Any)
// case class DoubleParam(value: Double) extends Param
// case class StringParam(value: String) extends Param
// case class IntParam(value: Int) extends Param

trait MainBase extends Constants{
    var defaultMap = mMap(
        'numVertices -> 1e7.toInt,
        'numEParts -> 320,
        'numVParts -> 320,
        'verbose -> 0,
        'noReverseGraph -> 0,
        'isCache -> 0,
        'run -> 0,
        'savePlain -> 0,
        'fillFeatures -> 0,
        'E2D -> 0, // 0: 1D, 1: 2D, 2: Random, 3: Can. Random
        'getReplication -> 0,
        'runPipeExp -> 0,
        // 'numWorkers -> 4,
        'debug -> 0,
        'numEpochs -> 1,
        'dataset -> 0,
        'miniBatchSize -> 5000,
        'naive -> 0,
        'aggPushDown -> 0,
        'selfLoop -> 0,
        'drillDown -> 0,
        'numLayers -> 2,
        'normalize -> 1,
        'sparse -> 0,
        'pipeGradMap -> 0,
        'ioType -> 0, // 0: raw string, 1: byte
        'numMachines -> 8,
        'CPUs -> 40,
        'hardPartition -> 0,
        'hardPartitionRead -> 0,
        'localityWaitZero -> 0,
        'ipcType -> 1, // 0: no shm, 1: shm
        'labeledInfo -> 0
    )
    val Pattern = "--([^ ]*)".r
    def parse(args: Array[String]): OptionMap = {
        val arglist = args.toList
        def nextOption(map: OptionMap, list: List[String]): OptionMap = {
            def isSwitch(s: String) = (s(0) == '-')

            list match {
                case Nil => map
                case "--numVertices" :: value :: tail =>
                    nextOption(
                        map ++ Map('numVertices -> value.toDouble.toInt),
                        tail
                    )
                case "--numEParts" :: value :: tail =>
                    nextOption(
                        map ++ Map('numEParts -> value.toDouble.toInt),
                        tail
                    )
                case "--numVParts" :: value :: tail =>
                    nextOption(
                        map ++ Map('numVParts -> value.toDouble.toInt),
                        tail
                    )
                case "--isCache" :: value :: tail =>
                    nextOption(
                        map ++ Map('isCache -> value.toDouble.toInt),
                        tail
                    )
                case "--verbose" :: value :: tail =>
                    nextOption(
                        map ++ Map('verbose -> value.toDouble.toInt),
                        tail
                    )
                case "--noReverseGraph" :: value :: tail =>
                    nextOption(
                        map ++ Map('noReverseGraph -> value.toDouble.toInt),
                        tail
                    )
                case "--run" :: value :: tail =>
                    nextOption(
                        map ++ Map('run -> value.toDouble.toInt),
                        tail
                    )
                case "--savePlain" :: value :: tail =>
                    nextOption(
                        map ++ Map('savePlain -> value.toDouble.toInt),
                        tail
                    )
                case "--fillFeatures" :: value :: tail =>
                    nextOption(
                        map ++ Map('fillFeatures -> value.toDouble.toInt),
                        tail
                    )
                case "--E2D" :: value :: tail =>
                    nextOption(
                        map ++ Map('E2D -> value.toDouble.toInt),
                        tail
                    )
                case "--getReplication" :: value :: tail =>
                    nextOption(
                        map ++ Map('getReplication -> value.toDouble.toInt),
                        tail
                    )
                // case "--numWorkers" :: value :: tail =>
                //     nextOption(
                //         map ++ Map('numWorkers -> value.toDouble.toInt),
                //         tail
                //     )
                case "--debug" :: value :: tail =>
                    nextOption(
                        map ++ Map('debug -> value.toDouble.toInt),
                        tail
                    )
                case "--numEpochs" :: value :: tail =>
                    nextOption(
                        map ++ Map('numEpochs -> value.toDouble.toInt),
                        tail
                    )
                // dataset:
                // 0: lognormal random graph
                // 1: ogbn-products
                // 2: ogbn-arxiv
                case "--dataset" :: value :: tail =>
                    nextOption(
                        map ++ Map('dataset -> value.toDouble.toInt),
                        tail
                    )
                case "--miniBatchSize" :: value :: tail =>
                    nextOption(
                        map ++ Map('miniBatchSize -> value.toDouble.toInt),
                        tail
                    )
                case "--naive" :: value :: tail =>
                    nextOption(
                        map ++ Map('naive -> value.toDouble.toInt),
                        tail
                    )
                case "--aggPushDown" :: value :: tail =>
                    nextOption(
                        map ++ Map('aggPushDown -> value.toDouble.toInt),
                        tail
                    )
                case "--selfLoop" :: value :: tail =>
                    nextOption(
                        map ++ Map('selfLoop -> value.toDouble.toInt),
                        tail
                    )
                case "--drillDown" :: value :: tail =>
                    nextOption(
                        map ++ Map('drillDown -> value.toDouble.toInt),
                        tail
                    )
                case "--numLayers" :: value :: tail =>
                    nextOption(
                        map ++ Map('numLayers -> value.toDouble.toInt),
                        tail
                    )
                case "--normalize" :: value :: tail =>
                    nextOption(
                        map ++ Map('normalize -> value.toDouble.toInt),
                        tail
                    )
                case "--sparse" :: value :: tail =>
                    nextOption(
                        map ++ Map('sparse -> value.toDouble.toInt),
                        tail
                    )
                case "--pipeGradMap" :: value :: tail =>
                    nextOption(
                        map ++ Map('pipeGradMap -> value.toDouble.toInt),
                        tail
                    )
                case "--ioType" :: value :: tail =>
                    nextOption(
                        map ++ Map('ioType -> value.toDouble.toInt),
                        tail
                    )
                case "--numMachines" :: value :: tail =>
                    nextOption(
                        map ++ Map('numMachines -> value.toDouble.toInt),
                        tail
                    )
                case "--CPUs" :: value :: tail =>
                    nextOption(
                        map ++ Map('CPUs -> value.toDouble.toInt),
                        tail
                    )
                case "--hardPartitionRead" :: value :: tail =>
                    nextOption(
                        map ++ Map('hardPartitionRead -> value.toDouble.toInt),
                        tail
                    )
                case "--hardPartition" :: value :: tail =>
                    nextOption(
                        map ++ Map('hardPartition -> value.toDouble.toInt),
                        tail
                    )
                case "--localityWaitZero" :: value :: tail =>
                    nextOption(
                        map ++ Map('localityWaitZero -> value.toDouble.toInt),
                        tail
                    )
                case "--labeledInfo" :: value :: tail =>
                    nextOption(
                        map ++ Map('labeledInfo -> value.toDouble.toInt),
                        tail
                    )
                case Pattern(c) :: value :: tail =>
                    nextOption(
                        map ++ Map(Symbol(c) -> value.toDouble.toInt),
                        tail
                    )
                // case string :: opt2 :: tail if isSwitch(opt2) =>
                //     nextOption(map ++ Map('infile -> string), list.tail)
                // case string :: Nil =>
                //     nextOption(map ++ Map('infile -> string), list.tail)
                case option :: tail =>
                    println("Unknown option " + option)
                    sys.exit(1)
            }
        }
        val options = nextOption(defaultMap, arglist)
        println(options)
        return options
    }
}

