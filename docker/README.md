# Use the container (docker ≥ 19.03 required)

To build:
```shell
cd docker/
docker build -t direct:latest .
```

To run using all GPUs:
```shell
docker run --gpus all -it \
	--shm-size=24gb --volume=<source_to_data>:/data --volume=<source_to_results>:/output\
	--name=direct direct:latest /bin/bash
```
