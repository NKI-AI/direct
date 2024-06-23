ARG CUDA="11.3.0"
ARG PYTORCH="1.10"
ARG PYTHON="3.8"

# TODO: conda installs its own version of cuda
FROM nvidia/cuda:${CUDA}-devel-ubuntu18.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

# libsm6 and libxext6 are needed for cv2
RUN apt-get -qq update && apt-get install -y --no-install-recommends libxext6 libsm6 libxrender1 \
    build-essential sudo libgl1-mesa-glx git wget rsync tmux nano dcmtk fftw3-dev liblapacke-dev \
    libpng-dev libopenblas-dev jq \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

RUN git clone https://github.com/mrirecon/bart.git /tmp/bart \
    && cd /tmp/bart \
    && make -j4 \
    && make install \
    && rm -rf /tmp/bart

# Make a user
# Rename /home to /users to prevent issues with singularity
RUN mkdir /users \
    && adduser --disabled-password --gecos '' --home /users/direct direct \
    && adduser direct sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER direct

WORKDIR /tmp
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH "/users/direct/miniconda3/bin:/tmp/bart/:$PATH:$CUDA_ROOT"

# Setup python packages
RUN conda update -n base conda -yq \
 && conda install python=${PYTHON} \
 && conda install jupyter \
 && conda install cudatoolkit=${CUDA} torchvision -c pytorch

RUN if [ "nightly$PYTORCH" = "nightly" ] ; then echo "Installing pytorch nightly" && \
    conda install pytorch -c pytorch-nightly; else conda install pytorch=${PYTORCH} -c pytorch ; fi

USER root
RUN mkdir direct:direct /direct && chown direct:direct /direct && chmod 777 /direct

USER direct

RUN jupyter notebook --generate-config
ENV CONFIG_PATH "/users/direct/.jupyter/jupyter_notebook_config.py"
COPY "docker/jupyter_notebook_config.py" ${CONFIG_PATH}

# Copy files into the docker
COPY [".", "/direct"]
WORKDIR /direct
USER root
RUN python -m pip install -e ".[dev]"
USER direct

ENV PYTHONPATH /tmp/bart/python:/direct

# Provide an open entrypoint for the docker
ENTRYPOINT $0 $@
