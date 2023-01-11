# Lotan

The code and artifacts for our paper *Lotan: Bridging the Gap between GNNs and Scalable Graph Analytics Engines*.

## Prerequisites

Lotan is a distributed system working on top of Apache Spark and PyTorch DDP. Therefore, Spark `>= 3.2.0` and HDFS must be installed and enabled. We only tested Spark with 3.2.0 version, and compatibilities with other versions are unknown.

It is highly recommended to put this project folder in an NFS or other shared filesystem accessible from the entire cluster. Further, key-less SSH must be set up in the cluster. There are a few good guides, for instance, [this one] (https://kb.rice.edu/page.php?id=108596).

On the other hand, a `python >= 3.8` environment needs to be set up on every node; you can use the following:

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
pip install dgl-cu113==0.8.2.post1 dglgo -f https://data.dgl.ai/wheels/repo.html
```

Also, if you do not already have Scala and SBT, use the below.

```bash
# Sdk and scala
curl -s "https://get.sdkman.io" | bash
source "/home/projectadmin/.sdkman/bin/sdkman-init.sh"
sdk install scala 2.12.15
sdk install sbt
```

Last, modify the relevant constants in `runner_helper.sh`:

```bash
HOSTS=<path to a hosts file contains all worker nodes ip addresses, one ip per line>
HOSTS_ALL=<same to above, except also with the master node ip>
master_ip=<master node ip address>
LOG_DIR="<log root directory, preferably on a NFS>/$TIMESTAMP"
MODEL_DIR="<model checkpoint directory, preferably on a NFS>/$TIMESTAMP"
```


## Setup
The Lotan project comes in two parts: one part is in Scala using GraphX, and the other part is in Python using PyTorch. The Python part needs to be installed, and the Scala part needs to be compiled. 

Install the python package:

```bash
pip install -e .
```

Compile the scala lib to .jar (this step is included in the bash script below so you don't have to do this):

```bash
cd graphp
# install sbt for scala first
sbt assembly
```

## Usage

Configure a few system parameters in `run_mb.sh`: 

```bash
spark_worker_cores=<each cluster node's number of CPU cores>
numEParts=<number of edge partitions in graphx>
numVParts=<number of vertex partitions>
numMachines=<number of machines in the cluster>
```
Modify the GNN hyperparameter search grid if you want at `gsys/constants.py:65`; at the moment, it only supports optimizer in `{adam, adagrad}`, learning rate, and drop-out rate.

Set alias to `python`:

```bash
export DGL_PY=python
```

To launch a Lotan training task, use `run_example.sh` script to run both the GraphX job and PyTorch job. 