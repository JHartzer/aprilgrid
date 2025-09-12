FROM ubuntu:24.04

# Install system packages
RUN apt-get update && apt-get -y install \
    build-essential \
    clang-15 \
    clangd \
    cmake \
    doxygen \
    git \
    graphviz \
    libgtest-dev \
    libopencv-dev \
    uncrustify \
    && rm -rf /var/lib/apt/lists/*
