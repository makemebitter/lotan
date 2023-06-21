# Lotan

The code and artifacts for our paper *Lotan: Bridging the Gap between GNNs and Scalable Graph Analytics Engines*.

## Prerequisites

**Spark and HDFS:** Lotan is a distributed system working on top of Apache Spark and PyTorch DDP. Therefore, Spark `>= 3.2.0` and HDFS must be installed and enabled. We only tested Spark with 3.2.0 version, and compatibilities with other versions are unknown.

**NFS and key-less SSH:** It is highly recommended to put this project folder in an NFS or other shared filesystem accessible from the entire cluster. Further, key-less SSH must be set up in the cluster. There are a few good guides, for instance, [this one](https://kb.rice.edu/page.php?id=108596).

**Python, Java, and Scala:** On the other hand, a `python >= 3.8`  along with `jdk == 8` and `scala == 2.12` environment needs to be set up on every node; you can use the following:

```bash
# install all debian packages
bash setup.sh
# install all prepherial pythonlibs
pip install -r requirements_master.txt

# pytorch, modify according to your cuda version
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113


# ogb
pip install ogb
# torch-scatter and torch-sparse, for DGL
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cu113.html

# dgl
pip install dgl-cu113==0.9.1.post1 dglgo -f https://data.dgl.ai/wheels/repo.html
```

Also, if you do not already have Scala and SBT, use the below.

```bash
# Sdk and scala
curl -s "https://get.sdkman.io" | bash
source "/home/projectadmin/.sdkman/bin/sdkman-init.sh"
sdk install scala 2.12.15
sdk install sbt
```

**Important global vars**:

Lotan requires each machine to be labeled with global variables; you need to run the following to give each host a permanent name and number:

```bash
echo "WORKER_NUMBER=<number you give, eg: 1>" | sudo tee -a /etc/environment
echo "WORKER_NAME=<name you give, eg: worker1>" | sudo tee -a /etc/environment
source /etc/environment
```

Last, modify the relevant constants in `runner_helper.sh`:

```bash
HOSTS=<path to a hosts file contains all worker nodes ip addresses, one ip per line>
# the file should look like
10.0.1.2
10.0.1.3
...

HOSTS_ALL=<same to above, except also with the master node ip>
# the file should look like
10.0.1.1
10.0.1.2
10.0.1.3
...

master_ip=<master node ip address>

LOG_DIR="<log root directory, preferably on a NFS>/$TIMESTAMP"

MODEL_DIR="<model checkpoint directory, preferably on a NFS>/$TIMESTAMP"
```

Also modify the hardcoded hosts located in `graphp/src/main/scala/Constants`:

```scala
// Modify this to the list of IP addresses of your workers
val hosts = Seq(
        "10.10.1.1",
        "10.10.1.2",
        "10.10.1.3",
        "10.10.1.4",
        "10.10.1.5",
        "10.10.1.6",
        "10.10.1.7",
        "10.10.1.8"
    )

// modify this to be your spark master's ip and port
val master = "spark://10.10.1.1:7077"
```



## Installation

The Lotan project comes in two parts: one part is in Scala using GraphX, and the other part is in Python using PyTorch. The Python part needs to be installed, and the Scala part needs to be compiled. 

**Python part:** Install the python package:

```bash
pip install -e .
```

**Scala part:** Compile the scala lib to .jar (this step is included in the bash script below, so you don't have to do this):

```bash
cd graphp
# install sbt for scala first
sbt assembly
```

## Usage

### Set system parameters

Configure a few system parameters in `run_mb.sh`: 

```bash
spark_worker_cores=<each cluster node's number of CPU cores>
numEParts=<number of edge partitions in graphx>
numVParts=<number of vertex partitions>
numMachines=<number of machines in the cluster>
```
Set alias to `python`:

```bash
export DGL_PY=python
```

### Set model hyperparameters
Modify the GNN hyperparameter search grid if you want at `gsys/constants.py:65`; at the moment, it only supports optimizer in `{adam, adagrad}`, learning rate, and drop-out rate. The models implemented include GCN and GIN. More models can be implemented under the same framework, see `gsys/nn.py` for examples.


### Quick start: run model training
To launch a Lotan training task, use `run_example.sh` script from your master node to run both the GraphX job and PyTorch job. 

```bash
bash run_example.sh
```
This will train a 3-layer GCN model on the OGBN-arxiv dataset. More advanced usages are in the comments.

Running this command will automatically set up the python workers and messengers on every worker host. The prompt will be switched to Spark. This will also print out a command like

```bash
tail -f /mnt/nfs/logs/run_logs/xxx/server/$WORKER_NAME.log
```

Run this command on every host to monitor the model's training info in real-time.

Full usage of the `run_mb.sh` script:

```bash
bash run_mb.sh <optional log root> <num of epochs to train> <num of workers> <model-related configs> <number of model layers>

For the <model-related configs>, refer to gsys/all_args.py. These configs include 
```

### Use your own dataset

You can run Lotan on your own dataset as long as you can convert it into the DGL format. For details, refer to `dgl_to_spark_data.py` and follow the code path there. It is fairly simple as you only need to plug the DGL object into the `read_node_dataset` in `gsys/data.py`.


### Terminate processes

Sometimes the processes may not gracefully exit, to terminate all of them, run the following on every host:

```bash
pkill -f server_main.py; pkill -f pipe.py
```

