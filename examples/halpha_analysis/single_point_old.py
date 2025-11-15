from despotic import cloud
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
from despotic.chemistry import NL99, NL99_GC, GOW
from despotic import cloud
from tqdm import tqdm 
import time
import config as cfg
# import yt


cell = cloud()

cell.Tg = 4082
cell.nH = 562.341325190349
cell.colDen = 1.7782794100389228e+23


co_line_map = []

# --- 設定所有常數 ---

cell.sigmaNT = 2.0e5
cell.comp.xoH2 = 0.1
cell.comp.xpH2 = 0.4
cell.comp.xHe = 0.1


cell.dust.alphaGD   = 3.2e-34    # Dust-gas coupling coefficient
cell.dust.sigma10   = 2.0e-25    # Cross section for 10K thermal radiation
cell.dust.sigmaPE   = 1.0e-21    # Cross section for photoelectric heating
cell.dust.sigmaISRF = 3.0e-22    # Cross section to the ISRF
cell.dust.beta      = 2.0        # Dust spectral index
cell.dust.Zd        = 1.0        # Abundance relative to Milky Way
cell.Td             = 10         # Dust temperature
cell.rad.TCMB       = 2.73       # CMB temperature
cell.rad.TradDust   = 0.0        # IR radiation field seen by the dust
cell.rad.ionRate    = 2.0e-17    # Primary ionization rate
cell.rad.chi        = 1.0        # ISRF normalized to Solar neighborhood

start_time = time.time()  # 2. 記錄開始時間

cell.comp.computeDerived(cell.nH)

# --- 執行核心計算 ---
print("-------set Temp Eq --------")
cell.setTempEq()
print(f"mu = {cell.comp.mu}")
print(f"Tg = {cell.Tg}")
print(f"Td = {cell.Td}")
print("----------------------------")

base_guesses = [
    cell.Tg
]

def _append_guess(target: list[float], value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        return
    for existing in target:
        if abs(value - existing) / max(existing, 1.0) < 0.05:
            return
    target.append(float(value))

guess_queue: list[float] = []
for guess in base_guesses:
    _append_guess(guess_queue, guess)

attempt_log: list[dict[str, Union[float, bool]]] = []
extra_guess_limit = 5
final_converged = False
final_Tg = float("nan")

while guess_queue:
    current_guess = guess_queue.pop(0)
    cell.Tg = current_guess
    attempt_start = time.time()
    
    stages = [
        {"maxTime": 1e16, "maxTempIter": 50},
        {"maxTime": 1e17, "maxTempIter": 100},
        {"maxTime": 1e18, "maxTempIter": 200},
        {"maxTime": 1e19, "maxTempIter": 400},
    ]
    converged = False
    last_stage = stages[0]
    for stage_idx, stage in enumerate(stages, start=1):
        last_stage = stage
        converged = cell.setChemEq(
            network=NL99,
            evolveTemp="iterate",
            tol=1e-6,
            maxTime=stage["maxTime"],
            maxTempIter=stage["maxTempIter"],
        )
        print(
            f"    stage {stage_idx}: maxTime={stage['maxTime']:.1e}, "
            f"maxTempIter={stage['maxTempIter']} -> converged={converged}"
        )
        if converged:
            break
    duration = time.time() - attempt_start
    final_Tg = cell.Tg
    attempt_log.append(
        {
            "guess": current_guess,
            "final_Tg": final_Tg,
            "converged": converged,
            "duration_s": duration,
        }
    )
    print(
        f"[Attempt {len(attempt_log):02d}] guess={current_guess:.3g} K | "
        f"final={final_Tg:.3g} K | converged={converged} | time={duration:.2f}s"
    )
    if converged:
        final_converged = True
        break
    if len(guess_queue) < extra_guess_limit:
        _append_guess(guess_queue, final_Tg)

print(f"converge = {final_converged}")
print(f"final Tg = {final_Tg}")

# --- 處理計算結果 ---
lines = cell.lineLum("CO")
co_int_TB = lines[0]["intTB"]
co_line_map.append(co_int_TB)

end_time = time.time()  # 3. 記錄結束時間

elapsed_time = end_time - start_time
print(f"程式運算時間: {elapsed_time:.4f} 秒")

print(f"co_line_map = {co_line_map}")
print(f"Tg final = {final_Tg}")
lines = cell.lineLum("C+")
cp_int_TB = lines[0]["intTB"]
print(f"C+_line_map = {cp_int_TB}")



# NL99_GC
# co_line_map = [np.float64(26.647168610309297)]
# Tg final = 5.131135278673986


 #GOW
# co_line_map = [np.float64(27.130430537040976)]
# Tg final = 5.177461694980736

# co_line_map = [np.float64(27.124421626015074)]
# Tg final = 5.176960879841981
