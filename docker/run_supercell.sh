#!/bin/bash
# portUrb supercell の起動ラッパー(sim-hub コード URB)。
#
# supercell は初期サウンディングを「カレントディレクトリからの相対パス」
# ./inputs/wrf_supercell_sounding.yaml で読む(custom_modules/sc_init.h)。
# ハブは作業フォルダ(/work)だけをマウントしてそこをカレントにするので、
# イメージ内の inputs/ を先に写してから本体を exec する。
# cp -n なので、再実行しても既にあるものは上書きしない。
#
# 使い方: run_supercell.sh input_supercell.yaml input_dycore_supercell.yaml
set -euo pipefail
if [ ! -f inputs/wrf_supercell_sounding.yaml ]; then
  mkdir -p inputs
  cp -n /app/experiments/examples/inputs/wrf_supercell_sounding.yaml inputs/
fi
exec /app/build/supercell "$@"
