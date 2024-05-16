FROM ubuntu:24.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install autoconf make wget gcc git -y && \
    wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.04.12/riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz && \
    tar -xzvf riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz && \
    rm -rf riscv64-elf-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz
