"""ハブ(sim-hub)との接点。URB = portUrb(ORNL)の理想化スーパーセル(WK1982)実験。

契約は vault `Claude Code 運用/SIM_UI_DOCKER_RULES.md` §3。4関数だけを外に見せる。
どの関数も例外を投げない。読めないものは飛ばして、部分的な結果を返す。
"""
from __future__ import annotations

import json
import pathlib
import re

ADAPTER_API_VERSION = 1

# 自作で復元したdycore設定(cs=350, buoy_theta=true, rsst=false)。
# 根拠: RTX3090実機検証で上流ソースのコメント・出力ファイル命名規則・
# 無効化されたCFL補正式の基準値(350)と一致することを確認済み(2026-09-24)。
_DYCORE_YAML = """cs: 350
buoy_theta: true
rsst: false
"""


def schema() -> dict:
    """画面はここから自動生成される。パラメータの実値をここ以外に散らかさない。"""
    return {
        "code": "URB",
        "title": "portUrb スーパーセル (WK1982)",
        "description": "ORNL製圧縮性力学コア(Kokkos/YAKL)によるWeisman-Klemp 1982"
                        "理想化スーパーセルLES。GPU(CUDA)ビルド、倍精度、Morrison"
                        "2モーメント微物理。",
        "params": [
            {"name": "nx_glob", "label": "格子数(x,y共通)", "type": "int",
             "default": 100, "min": 50, "max": 500, "unit": ""},
            {"name": "xlen_km", "label": "領域一辺(km、x,y共通)", "type": "float",
             "default": 200.0, "min": 20.0, "max": 200.0, "unit": "km"},
            {"name": "sim_time_s", "label": "積分時間", "type": "float",
             "default": 120.0, "min": 60.0, "max": 7200.0, "unit": "s"},
            {"name": "out_freq_s", "label": "出力間隔(負で出力オフ)", "type": "float",
             "default": -1.0, "min": -1.0, "max": 3600.0, "unit": "s"},
        ],
        "presets": {
            "quick": {"label": "下見(100x100x50, 2km格子, 120秒)",
                      "values": {"nx_glob": 100, "xlen_km": 200.0,
                                 "sim_time_s": 120.0, "out_freq_s": -1.0}},
            "standard": {"label": "本番相当(500x500x50, 400m格子, 2時間)※数時間かかる",
                         "values": {"nx_glob": 500, "xlen_km": 200.0,
                                    "sim_time_s": 7200.0, "out_freq_s": 360.0}},
        },
        "estimate_hint": "格子点数(nx_glob^2 * 50) x 積分時間に比例して重くなる",
    }


def estimate_seconds(params: dict) -> float:
    """RTX3090実機較正(400m/500x500x50/120sで314秒)から単純外挿。目安のみ。"""
    try:
        nx = float(params.get("nx_glob", 100))
        t = float(params.get("sim_time_s", 120.0))
        ref_pts = 500.0 * 500.0 * 50.0
        ref_time = 314.4  # 秒(実測較正: 120sモデル秒 @ 500x500x50)
        ref_sim_s = 120.0
        pts = nx * nx * 50.0
        return ref_time * (pts / ref_pts) * (t / ref_sim_s)
    except Exception:
        return float("nan")


def prepare(params: dict, run_dir) -> dict:
    """input_supercell.yaml / input_dycore_supercell.yaml を run_dir に書く。

    スレッド数をここで決めない。ハブが1コア空けた数を OMP_NUM_THREADS 等で渡す。
    """
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    nx_glob = int(params.get("nx_glob", 100))
    xlen = float(params.get("xlen_km", 200.0)) * 1000.0
    sim_time = float(params.get("sim_time_s", 120.0))
    out_freq = float(params.get("out_freq_s", -1.0))

    # 自分用の設定は params.json とは別名で書く(ハブが params.json を上書きするため)
    (run_dir / "urb_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    supercell_yaml = f"""---
sim_time: {sim_time}
xlen: {xlen:.0f}
ylen: {xlen:.0f}
zlen: 20000
nx_glob: {nx_glob}
ny_glob: {nx_glob}
nz     : 50
dt_phys: 0.
out_freq: {out_freq}
inform_freq: 10.
is_restart: false
restart_file: none
cfl: 0.6
"""
    (run_dir / "input_supercell.yaml").write_text(supercell_yaml, encoding="utf-8")
    (run_dir / "input_dycore_supercell.yaml").write_text(_DYCORE_YAML, encoding="utf-8")

    return {
        "command": ["/app/build/supercell",
                    "/work/input_supercell.yaml",
                    "/work/input_dycore_supercell.yaml"],
        "workdir": "/work",
        "needs": [],
        "env": {},
        "ulimits": {},
    }


def progress(run_dir) -> dict:
    """run.log を読むだけ。プロセスは見ない。"""
    state = {"percent": 0.0, "message": "まだログがありません(起動待ち)",
             "finished": False, "failed": False}
    try:
        text = (pathlib.Path(run_dir) / "run.log").read_text(
            encoding="utf-8", errors="ignore")
    except OSError:
        return state

    if "terminate called" in text or "Signal:" in text or "std::logic_error" in text:
        state.update(percent=100.0, message="異常終了しました", failed=True)
        return state

    # 実際の標準出力書式(RTX3090実機で確認済み):
    #   MaxWind [...] , Etime [1.234560e+02 s] , Walltime [...] , max wind [...] , max(abs(w)) [...]
    # portUrb自体は完了を示す文字列を出さないため、Etimeがsim_timeに達したことで判定する。
    etimes = re.findall(r"Etime\s*\[\s*([0-9.eE+-]+)\s*s\s*\]", text)
    try:
        params = json.loads((pathlib.Path(run_dir) / "urb_params.json").read_text())
        target = float(params.get("sim_time_s", 120.0))
    except Exception:
        target = None

    if etimes:
        sim_t = float(etimes[-1])
        if target and target > 0:
            pct = min(100.0, 100.0 * sim_t / target)
            state.update(percent=pct, message=f"モデル時間 {sim_t:.0f}s / {target:.0f}s")
            if sim_t >= target - 1e-6:
                state.update(finished=True, message=f"完走しました(モデル時間 {sim_t:.0f}s)")
        else:
            state["message"] = f"モデル時間 {sim_t:.0f}s"

    return state


def results(run_dir) -> list[dict]:
    """[{"path", "caption", "kind"}]。caption は必須。無ければ空リスト。"""
    run_dir = pathlib.Path(run_dir)
    items = []
    try:
        for nc in sorted(run_dir.glob("supercell_*.nc")):
            items.append({
                "path": nc.name,
                "kind": "metrics",
                "caption": f"portUrb出力 ({nc.name})",
            })
    except OSError:
        pass
    figures = run_dir / "figures"
    if figures.is_dir():
        for png in sorted(figures.glob("*.png")):
            items.append({
                "path": f"figures/{png.name}",
                "kind": "png",
                "caption": "portUrb スーパーセル可視化",
            })
    return items
