import os
from gsys.all_args import get_all_args

if __name__ == "__main__":
    args = get_all_args()

    filename = "{}/conf/spark-env.sh".format(os.getenv("SPARK_HOME"))

    with open(filename, 'r') as f:
        lines = f.readlines()

    found = False
    substitute = "export SPARK_WORKER_CORES={}\n".format(
        args.spark_worker_cores)
    for i, x in enumerate(lines):
        if x.startswith("export SPARK_WORKER_CORES"):
            found = True
            lines[i] = substitute

    if not found:
        lines.append(substitute)

    with open(filename, 'w') as f:
        f.write(''.join(lines))
