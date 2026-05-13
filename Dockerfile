FROM precice/precice:nightly

RUN apt-get -qq update && apt-get -qq install -y \
    sudo \
    python3-dev \
    python3-venv \
    git \
    pkg-config \
    pybind11-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mpi4py

WORKDIR /micro-manager
