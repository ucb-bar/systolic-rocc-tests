FROM ubuntu:24.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH "$PATH:/riscv/bin"
ENV RISCV "/riscv"


RUN apt-get update && \
    apt-get install autoconf make wget gcc git device-tree-compiler build-essential -y && \
    wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.04.12/riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz && \
    tar -xzvf riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz && \
    rm -rf riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz && \
    git clone https://github.com/riscv-software-src/riscv-isa-sim.git && \
    cd riscv-isa-sim && \
    mkdir build && \
    cd build && \
    ../configure --prefix=$RISCV && \
    make -j$(nproc) && make install && \
    cd / && \
    git clone https://github.com/ucb-bar/libgemmini && \
    cd libgemmini && \
    make -j$(nproc) && make install

RUN apt-get install -y lsb-release wget software-properties-common gnupg python3-pip && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 17

RUN apt-get install -y mlir-17-tools

RUN git clone https://github.com/KULeuven-MICAS/snax-mlir.git && \
    cd snax-mlir && \
    git checkout Josse/add-gemmini-loop-ws && \
    pip install pre-commit filecheck==1.0.0 lit "numpy<2.0.0" xdsl@git+https://github.com/xdslproject/xdsl.git@04bdfc0d3186d4451423eb308d7fea232ba77435 --break-system-packages && \
    pip install -e  . --break-system-packages
