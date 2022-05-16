Docker Installation
-------------------

Use the container
~~~~~~~~~~~~~~~~~

To build:

.. code-block:: bash

    cd direct/
    docker build -t direct:latest -f docker/Dockerfile .

To run using all GPUs:

.. code-block:: bash

    docker run --gpus all -it \
        --shm-size=24gb --volume=<source_to_data>:/data --volume=<source_to_results>:/output \
	    --name=direct direct:latest /bin/bash

Requirements
~~~~~~~~~~~~

* docker ≥ 19.03
