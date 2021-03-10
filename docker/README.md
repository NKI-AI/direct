# Use the container (docker ≥ 19.03 required)

To build:

```
cd docker/
docker build -t direct:latest .
```

if you want to use the nightly version of pytorch append `--build-arg PYTORCH=nightly` to the build command.

To run using all GPUs:

```
docker run --gpus all -it \
	--shm-size=24gb --volume=<source_to_data>:/data --volume=<source_to_results>:/output\
	--name=direct direct:latest /bin/bash
```
