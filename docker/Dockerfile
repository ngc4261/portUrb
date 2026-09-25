# portUrb (ORNL) supercell 実験 GPU ビルド。sim-hub コード: URB。
# ビルド文脈はリポジトリ直下(このファイルは docker/ に置くが `-f docker/Dockerfile .` で呼ぶ)。
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran cmake git ca-certificates \
        openmpi-bin libopenmpi-dev \
        libnetcdf-dev libnetcdff-dev libpnetcdf-dev \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# サブモジュールが git@github.com: (SSH) を指しているため、ビルド時は
# 認証情報を持たない匿名HTTPSへ読み替える(踏みやすい罠#: SSH鍵はコンテナに無い)
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

WORKDIR /app
COPY . /app

# model/CMakeLists.txt は kokkos・YAKL・yaml-cpp・ponni・eigen を add_subdirectory する。
# 一部だけ初期化すると ponni が空で configure が止まる(GitHub Actions run 36155354906 で実際に失敗)。
# WGS84toCartesian も含めて全部入れる(ヘッダのみ、小さい)。
RUN git submodule update --init --recursive

# aura26_gpu_O3.env と同じ設定(-O3、YAKL_PROFILE無し、Ampere 86 = RTX 3090/3080)。
# -march=native は使わない(踏みやすい罠: 組んだ機体でしか動かなくなる)。
ENV CC=mpicc \
    CXX=mpicxx \
    FC=mpifort \
    NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx \
    PORTURB_BACKEND=Kokkos_ENABLE_CUDA \
    PORTURB_ARCH=Kokkos_ARCH_AMPERE86 \
    PORTURB_CXX_FLAGS="-DHAVE_MPI;-O3" \
    PORTURB_F90_FLAGS="-cpp;-ffree-line-length-none;-O3;-fdefault-real-8;-fdefault-double-8" \
    PORTURB_LINK_FLAGS="-lpnetcdf;-lnetcdf"

WORKDIR /app/build
RUN ./cmakescript.sh ../experiments/examples \
    && make -j"$(nproc)" supercell

# 起動ラッパー。supercell は ./inputs/wrf_supercell_sounding.yaml をカレントから読むため、
# 作業フォルダ(/work)に inputs/ を写してから本体を exec する。
# ENTRYPOINT は付けない: sim-hub は `docker run <image> <command...>` の形で完全なコマンドを
# 渡す(needs があるときは `sh -c`)ので、ENTRYPOINT があると引数が二重になる。
COPY docker/run_supercell.sh /app/run_supercell.sh
RUN chmod +x /app/run_supercell.sh

# 実行時のスレッド数は sim-hub 側が OMP_NUM_THREADS 等で渡す(コード側で決め打ちしない)。
# 手で動かすとき: docker run --rm --gpus all -v "$PWD":/work -w /work <image> \
#                  /app/run_supercell.sh input_supercell.yaml input_dycore_supercell.yaml
CMD ["/app/run_supercell.sh", "input_supercell.yaml", "input_dycore_supercell.yaml"]
