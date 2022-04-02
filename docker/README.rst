Docker Installation
-------------------

Use the container (docker ≥ 19.03 required)
-------------------------------------------

To build:

.. code-block:: bash

    cd docker/
    docker build -t direct:latest .

To run using all GPUs:

.. code-block:: bash

    docker run --gpus all -it \
        --shm-size=24gb --volume=<source_to_data>:/data --volume=<source_to_results>:/output \
	    --name=direct direct:latest /bin/bash
