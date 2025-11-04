# -*- coding: utf-8 -*-
"""
右/左心系统功能指标（CT 多期体积 → 拟合曲线）
- 周期三次样条 + Savitzky–Golay 平滑
- 稳健事件识别（相位窗）：E/A 与最大射血速度（修正早期/边界伪峰）
- 右心三张图：volume_curve.png / dvdt_curve.png / ra_rv_coupling.png
- BSA（Mosteller或直接BSA列）指数化，兼容 high=身高、height=体重
- 输出：
  right_heart_metrics_filledIDs.csv / right_heart_metrics_indexed.csv
  left_heart_metrics_filledIDs.csv  / left_heart_metrics_indexed.csv
  right_heart_metrics_qc.csv
  每例一个文件夹（以影像号命名）保存图与 resampled_curves.csv
"""

import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks

# ======== 路径（按需修改） ========
INPUT_XLSX = r"D:\数据资料\联影电影数据\ex3.XLSX"
BSA_XLSX   = r"D:\数据资料\联影电影数据\BSA3.xlsx"
# ======== 参数 ========
UPSAMPLE_POINTS = 4000        # 重采样点数（越大越平滑）
SG_WINDOW_FRAC  = 0.02        # SG 窗口（相对M）
SG_POLY         = 3
PEAK_MIN_DISTANCE_PCT = 5.0   # 峰间最小距离（%cycle）
PEAK_MIN_PROM_STD     = 0.5   # 峰显著性（std倍数）
DPI = 150

# ---------- 基础工具 ----------
def safe_dirname(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s).replace(" ", "_")
    return s if s else "UNK"

def canonical_id(val) -> str:
    """统一影像号格式：去空格、去尾随'.0'，数值转整数字符串。"""
    if val is None or (isinstance(val, float) and not np.isfinite(val)): return ""
    try:
        if isinstance(val, (int, np.integer)): return str(int(val))
        if isinstance(val, (float, np.floating)):
            if np.isfinite(val) and float(val).is_integer(): return str(int(val))
            s = format(val, "f").rstrip("0").rstrip(".")
        else:
            s = str(val)
    except Exception:
        s = str(val)
    s = s.strip()
    if s.endswith(".0"): s = s[:-2]
    return s.replace(" ", "")

def _savgol(y: np.ndarray, win_frac: float, poly: int) -> np.ndarray:
    """SG 平滑（周期模式），窗口自动奇数与上限保护。"""
    n = len(y)
    w = max(7, int(n * win_frac))
    if w % 2 == 0: w += 1
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if w < poly + 2:
        w = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if w < 7: return y
    return savgol_filter(y, window_length=w, polyorder=poly, mode="wrap")

def resample_with_periodic_spline(vols: np.ndarray,
                                  upsample_M: int = UPSAMPLE_POINTS,
                                  sg_frac: float = SG_WINDOW_FRAC,
                                  sg_poly: int = SG_POLY) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """周期三次样条 + SG 平滑；若首尾不等则补 (1.0,y0)。返回 t(0~1)、y(t)、dy/dt(t)。"""
    vols = np.asarray(vols, dtype=float).copy()
    N = vols.shape[0]
    t_orig = np.linspace(0.0, 1.0, N, endpoint=False)

    # 填补 NaN/inf
    idx = np.arange(N)
    mask = np.isfinite(vols)
    if not mask.all() and mask.any():
        vols[~mask] = np.interp(idx[~mask], idx[mask], vols[mask])

    if not np.isclose(vols[0], vols[-1], rtol=1e-9, atol=1e-9):
        t_fit = np.concatenate([t_orig, [1.0]])
        y_fit = np.concatenate([vols,  [vols[0]]])
    else:
        t_fit = t_orig
        y_fit = vols

    cs = CubicSpline(t_fit, y_fit, bc_type="periodic")
    t_hr = np.linspace(0.0, 1.0, upsample_M, endpoint=False)
    y_hr  = cs(t_hr)
    dy_hr = cs(t_hr, 1)

    y_sm  = _savgol(y_hr,  sg_frac, sg_poly)
    dy_sm = _savgol(dy_hr, sg_frac, sg_poly)
    return t_hr, y_sm, dy_sm

def forward_interval(i_start: int, i_end: int, M: int) -> np.ndarray:
    """沿周期从 start→end 的索引序列（含端点）。"""
    return np.arange(i_start, i_end + 1) if i_start <= i_end else \
           np.concatenate([np.arange(i_start, M), np.arange(0, i_end + 1)])

# ---------- 稳健相位窗：进度/窗口化峰值 ----------
def _progress_in_arc(idx: int, start: int, end: int, M: int) -> float:
    """返回 idx 在 start→end 弧段的归一化位置 p∈[0,1]（含端点）。"""
    if start <= end:
        length = end - start + 1
        pos = idx - start
    else:
        length = (M - start) + (end + 1)
        pos = (idx - start) if idx >= start else (M - start) + idx
    return max(0.0, min(1.0, pos / max(1, length - 1)))

def find_EA_windowed(drv: np.ndarray, idx_ES: int, idx_ED: int, M: int,
                     e_win=(0.05, 0.35), a_win=(0.55, 0.95),
                     min_dist_pct=PEAK_MIN_DISTANCE_PCT, prom_std=PEAK_MIN_PROM_STD, edge_buffer=0.01):
    """
    在 ES→ED 段，仅在给定相位窗内找正峰：
      E 窗：默认 5%~35%，取最早峰；A 窗：默认 55%~95%，取最晚峰。
    无峰时使用最近窗中心的峰兜底；强制 A 晚于 E。
    """
    seg = forward_interval(idx_ES, idx_ED, M)
    min_distance = max(3, int(M * (min_dist_pct / 100.0)))
    prom = max(1e-6, float(np.nanstd(drv)) * prom_std)
    peaks, _ = find_peaks(drv, distance=min_distance, prominence=prom)

    seg_set = set(seg)
    peaks_in = []
    for p in peaks:
        if p in seg_set:
            pr = _progress_in_arc(p, idx_ES, idx_ED, M)
            if edge_buffer <= pr <= 1.0 - edge_buffer:
                peaks_in.append((p, pr))
    peaks_in.sort(key=lambda x: x[1])

    def _pick(win, earliest=True):
        lo, hi = win
        cands = [(p, pr) for (p, pr) in peaks_in if lo <= pr <= hi]
        if not cands: return None
        return cands[0][0] if earliest else cands[-1][0]

    iE = _pick(e_win, earliest=True)
    iA = _pick(a_win, earliest=False)

    if iE is None and peaks_in:
        center = np.mean(e_win)
        iE = min(peaks_in, key=lambda x: abs(x[1]-center))[0]
    if iA is None and peaks_in:
        if iE is not None:
            prE = _progress_in_arc(iE, idx_ES, idx_ED, M)
            late = [(p,pr) for (p,pr) in peaks_in if pr >= prE]
            if late:
                iA = late[-1][0]
        if iA is None:
            iA = max(peaks_in, key=lambda x: x[1])[0]
    return iE, iA

def find_eject_windowed(drv: np.ndarray, idx_ED: int, idx_ES: int, M: int,
                        ej_win=(0.05, 0.45), edge_buffer=0.01):
    """
    在 ED→ES 段，优先在 ej_win（默认 5%~45%）寻找最负 dV/dt。
    无则回退到除去边界的全段；再无则全段。
    """
    seg = forward_interval(idx_ED, idx_ES, M)
    prog = [(k, _progress_in_arc(k, idx_ED, idx_ES, M)) for k in seg]
    lo, hi = ej_win
    cand = [k for (k, pr) in prog if (lo + edge_buffer) <= pr <= (hi - edge_buffer)]
    if cand:
        return cand[int(np.argmin(drv[cand]))]
    cand2 = [k for (k, pr) in prog if edge_buffer <= pr <= 1.0 - edge_buffer]
    if cand2:
        return cand2[int(np.argmin(drv[cand2]))]
    return seg[int(np.argmin(drv[seg]))]

# ---------- BSA 读取（兼容 high=身高、height=体重） ----------
def load_bsa_table(bsa_path: str) -> Optional[pd.DataFrame]:
    if not bsa_path or not os.path.exists(bsa_path):
        print("未找到BSA文件，将跳过体积指数化。"); return None
    df = pd.read_excel(bsa_path, engine="openpyxl")

    # 影像号列
    id_col = None
    for c in df.columns:
        cstr = str(c).strip()
        if cstr in ["影像号","PatientID","CaseID","ID","编号"] or "影像" in cstr:
            id_col = c; break
    if id_col is None:
        print("BSA表缺少影像号列，跳过指数化。"); return None

    # 直接BSA
    bsa_col = None
    for cand in ["BSA","体表面积","BSA_m2","bsa"]:
        if cand in df.columns: bsa_col=cand; break

    out = pd.DataFrame()
    out["PatientID"] = df[id_col].apply(canonical_id)

    if bsa_col is not None and df[bsa_col].notna().any():
        out["BSA_m2"] = pd.to_numeric(df[bsa_col], errors="coerce")
    else:
        # 你的文件约定：high=身高，height=体重；缺失时再兜底
        h_col = "high"   if "high"   in df.columns else None
        w_col = "height" if "height" in df.columns else None
        if h_col is None:
            for cand in ["身高","身高(cm)","height_cm","height","Height"]:
                if cand in df.columns: h_col = cand; break
        if w_col is None:
            for cand in ["体重","体重(kg)","weight","weight_kg","Weight","height"]:
                if cand in df.columns: w_col = cand; break
        if h_col is None or w_col is None:
            print("BSA表缺少 BSA 或 身高/体重列，跳过指数化。"); return None

        height = pd.to_numeric(df[h_col], errors="coerce")  # cm 或 m
        weight = pd.to_numeric(df[w_col], errors="coerce")  # kg 或 g
        if height.dropna().median() < 3.0:   height = height * 100.0   # 米→厘米
        if weight.dropna().median() > 300.0: weight = weight / 1000.0  # 克→千克
        out["BSA_m2"] = np.sqrt(height * weight / 3600.0)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["BSA_m2"])
    out = out.groupby("PatientID", as_index=False)["BSA_m2"].first()
    return out

def add_bsa_indexing_generic(res_df: pd.DataFrame, bsa_df: Optional[pd.DataFrame],
                             vol_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合并 BSA 并添加 mL/m² 指标；返回 (merged, unmatched_ids)。"""
    if bsa_df is None or bsa_df.empty:
        return res_df.copy(), pd.DataFrame(columns=["PatientID"])
    df = res_df.copy()
    df["PatientID"] = df["PatientID"].apply(canonical_id)
    bsa_df = bsa_df.copy()
    bsa_df["PatientID"] = bsa_df["PatientID"].apply(canonical_id)
    merged = df.merge(bsa_df, on="PatientID", how="left", suffixes=("", "_bsa"))
    if "BSA_m2_bsa" in merged.columns:
        merged["BSA_m2"] = merged["BSA_m2_bsa"]; merged.drop(columns=["BSA_m2_bsa"], inplace=True)
    unmatched = merged[merged["BSA_m2"].isna()][["PatientID"]].drop_duplicates()

    def idx_div(x, bsa):
        try:
            return x / bsa if (bsa is not None and np.isfinite(bsa) and bsa > 0) else np.nan
        except Exception:
            return np.nan

    for src, dst in vol_map.items():
        if src in merged.columns:
            merged[dst] = [idx_div(v, b) for v, b in zip(merged[src].values, merged["BSA_m2"].values)]
    return merged, unmatched

# ---------- 指标计算：右心 ----------
def compute_right_metrics(patient_id: str,
                          chamber_vols: Dict[str, np.ndarray],
                          phase_names: List[str]) -> Dict[str, Optional[float]]:
    res = {"PatientID": canonical_id(patient_id), "N_phases": len(phase_names)}
    need = {"RA","RV","LA","LV"}
    if not need.issubset(chamber_vols.keys()):
        for k in ["RV_EDV","RV_ESV","RV_SV","RVEF_%",
                  "RV_max_eject_vel_ml_perc","t_RV_max_eject_%",
                  "RV_max_fill_E_vel_ml_perc","t_RV_E_%",
                  "RV_max_fill_A_vel_ml_perc","t_RV_A_%",
                  "RA_Vmax","RA_VpreA","RA_Vmin",
                  "RA_reservoir_EF_%","RA_passive_EF_%","RA_active_EF_%",
                  "RA_RV_coupling_total","RA_RV_coupling_active","RA_RV_SV_ratio",
                  "PhaseDiff_ED_RVminusLV_%","PhaseDiff_ES_RVminusLV_%",
                  "PeakTimeDiff_Eject_RVminusLV_%","PeakTimeDiff_FillE_RVminusLV_%"]:
            res[k] = np.nan
        return res

    t_RA, y_RA, d_RA = resample_with_periodic_spline(chamber_vols["RA"])
    t_RV, y_RV, d_RV = resample_with_periodic_spline(chamber_vols["RV"])
    t_LV, y_LV, d_LV = resample_with_periodic_spline(chamber_vols["LV"])
    t_LA, y_LA, d_LA = resample_with_periodic_spline(chamber_vols["LA"])
    M = len(t_RV)

    # RV ED/ES + EF
    i_RV_ED = int(np.argmax(y_RV)); i_RV_ES = int(np.argmin(y_RV))
    RV_EDV, RV_ESV = float(y_RV[i_RV_ED]), float(y_RV[i_RV_ES])
    RV_SV = RV_EDV - RV_ESV
    RVEF = (RV_SV / RV_EDV * 100.0) if RV_EDV > 0 else np.nan

    # 射血峰（相位窗）
    i_eject = find_eject_windowed(d_RV, i_RV_ED, i_RV_ES, M, ej_win=(0.05, 0.45))
    RV_max_eject_vel_ml_perc = float(d_RV[i_eject]) / 100.0
    t_RV_eject_pct = i_eject / M * 100.0

    # E/A（相位窗）
    i_E, i_A = find_EA_windowed(d_RV, i_RV_ES, i_RV_ED, M,
                                e_win=(0.05, 0.35), a_win=(0.55, 0.95))
    RV_E_vel_ml_perc = float(d_RV[i_E]) / 100.0; t_RV_E_pct = i_E / M * 100.0
    if i_A is not None:
        RV_A_vel_ml_perc = float(d_RV[i_A]) / 100.0; t_RV_A_pct = i_A / M * 100.0
    else:
        RV_A_vel_ml_perc, t_RV_A_pct = np.nan, np.nan

    # RA 三相
    i_RA_Vmax, i_RA_Vmin = int(np.argmax(y_RA)), int(np.argmin(y_RA))
    RA_Vmax, RA_Vmin = float(y_RA[i_RA_Vmax]), float(y_RA[i_RA_Vmin])
    idx_fill = forward_interval(i_RV_ES, i_RV_ED, M)
    if i_A is not None:
        i_target = i_A
    else:
        frac_idx = int(0.8 * (len(idx_fill) - 1))
        i_target = idx_fill[frac_idx] if len(idx_fill)>0 else (i_RA_Vmin + int(0.1*M))%M
    RA_VpreA = float(y_RA[i_target])

    RA_total_empty = RA_Vmax - RA_Vmin
    RA_reservoir_EF = (RA_total_empty / RA_Vmax * 100.0) if RA_Vmax>0 else np.nan
    RA_passive_EF   = ((RA_Vmax - RA_VpreA) / RA_Vmax * 100.0) if RA_Vmax>0 else np.nan
    RA_active_EF    = ((RA_VpreA - RA_Vmin) / RA_VpreA * 100.0) if RA_VpreA>0 else np.nan

    # RV-LV 相位差与峰值时间差
    i_LV_ED, i_LV_ES = int(np.argmax(y_LV)), int(np.argmin(y_LV))
    def wrap_pct(delta):
        while delta <= -50.0: delta += 100.0
        while delta > 50.0: delta -= 100.0
        return delta
    phase_ED = wrap_pct((i_RV_ED - i_LV_ED) / M * 100.0)
    phase_ES = wrap_pct((i_RV_ES - i_LV_ES) / M * 100.0)
    i_lv_eject = find_eject_windowed(d_LV, i_LV_ED, i_LV_ES, M, ej_win=(0.05, 0.45))
    peak_eject_diff = wrap_pct((i_eject - i_lv_eject) / M * 100.0)
    i_lv_E, _ = find_EA_windowed(d_LV, i_LV_ES, i_LV_ED, M,
                                 e_win=(0.05, 0.35), a_win=(0.55, 0.95))
    peak_fillE_diff = wrap_pct((i_E - i_lv_E) / M * 100.0)

    res.update({
        "RV_EDV": RV_EDV, "RV_ESV": RV_ESV, "RV_SV": RV_SV, "RVEF_%": RVEF,
        "RV_max_eject_vel_ml_perc": RV_max_eject_vel_ml_perc, "t_RV_max_eject_%": t_RV_eject_pct,
        "RV_max_fill_E_vel_ml_perc": RV_E_vel_ml_perc, "t_RV_E_%": t_RV_E_pct,
        "RV_max_fill_A_vel_ml_perc": RV_A_vel_ml_perc, "t_RV_A_%": t_RV_A_pct,
        "RA_Vmax": RA_Vmax, "RA_VpreA": RA_VpreA, "RA_Vmin": RA_Vmin,
        "RA_reservoir_EF_%": RA_reservoir_EF, "RA_passive_EF_%": RA_passive_EF, "RA_active_EF_%": RA_active_EF,
        "RA_RV_coupling_total": (RA_reservoir_EF/RVEF) if (RVEF and not np.isnan(RVEF)) else np.nan,
        "RA_RV_coupling_active": (RA_active_EF/RVEF) if (RVEF and not np.isnan(RVEF)) else np.nan,
        "RA_RV_SV_ratio": (RA_total_empty/RV_SV) if RV_SV != 0 else np.nan,
        "PhaseDiff_ED_RVminusLV_%": phase_ED, "PhaseDiff_ES_RVminusLV_%": phase_ES,
        "PeakTimeDiff_Eject_RVminusLV_%": peak_eject_diff,
        "PeakTimeDiff_FillE_RVminusLV_%": peak_fillE_diff,
        # 用于绘图
        "_t_pct": np.linspace(0, 100, M, endpoint=False),
        "_y_LA": y_LA, "_y_LV": y_LV, "_y_RA": y_RA, "_y_RV": y_RV,
        "_d_LA": d_LA/100.0, "_d_RA": d_RA/100.0, "_d_LV": d_LV/100.0, "_d_RV": d_RV/100.0,
        "_idx": {"RV_ED": i_RV_ED, "RV_ES": i_RV_ES, "RV_E": i_E, "RV_A": (i_A if i_A is not None else None),
                 "LV_ED": i_LV_ED, "LV_ES": i_LV_ES, "LV_E": i_lv_E, "RV_eject": i_eject, "LV_eject": i_lv_eject}
    })
    return res

# ---------- 指标计算：左心 ----------
def compute_left_metrics(patient_id: str,
                         chamber_vols: Dict[str, np.ndarray],
                         phase_names: List[str]) -> Dict[str, Optional[float]]:
    res = {"PatientID": canonical_id(patient_id), "N_phases": len(phase_names)}
    need = {"LA","LV"}
    if not need.issubset(chamber_vols.keys()):
        for k in ["LV_EDV","LV_ESV","LV_SV","LVEF_%",
                  "LV_max_eject_vel_ml_perc","t_LV_max_eject_%",
                  "LV_max_fill_E_vel_ml_perc","t_LV_E_%",
                  "LV_max_fill_A_vel_ml_perc","t_LV_A_%",
                  "LA_Vmax","LA_VpreA","LA_Vmin",
                  "LA_reservoir_EF_%","LA_passive_EF_%","LA_active_EF_%",
                  "LA_LV_coupling_total","LA_LV_coupling_active","LA_LV_SV_ratio"]:
            res[k] = np.nan
        return res

    t_LA, y_LA, d_LA = resample_with_periodic_spline(chamber_vols["LA"])
    t_LV, y_LV, d_LV = resample_with_periodic_spline(chamber_vols["LV"])
    M = len(t_LV)

    # LV ED/ES + EF
    i_LV_ED = int(np.argmax(y_LV)); i_LV_ES = int(np.argmin(y_LV))
    LV_EDV, LV_ESV = float(y_LV[i_LV_ED]), float(y_LV[i_LV_ES])
    LV_SV = LV_EDV - LV_ESV
    LVEF = (LV_SV / LV_EDV * 100.0) if LV_EDV > 0 else np.nan

    # 射血峰（相位窗）
    i_eject = find_eject_windowed(d_LV, i_LV_ED, i_LV_ES, M, ej_win=(0.05, 0.45))
    LV_max_eject_vel_ml_perc = float(d_LV[i_eject]) / 100.0
    t_LV_eject_pct = i_eject / M * 100.0

    # E/A（相位窗）
    i_E, i_A = find_EA_windowed(d_LV, i_LV_ES, i_LV_ED, M,
                                e_win=(0.05, 0.35), a_win=(0.55, 0.95))
    LV_E_vel_ml_perc = float(d_LV[i_E]) / 100.0; t_LV_E_pct = i_E / M * 100.0
    if i_A is not None:
        LV_A_vel_ml_perc = float(d_LV[i_A]) / 100.0; t_LV_A_pct = i_A / M * 100.0
        has_A = True
    else:
        LV_A_vel_ml_perc, t_LV_A_pct, has_A = np.nan, np.nan, False

    # LA 三相
    i_LA_Vmax, i_LA_Vmin = int(np.argmax(y_LA)), int(np.argmin(y_LA))
    LA_Vmax, LA_Vmin = float(y_LA[i_LA_Vmax]), float(y_LA[i_LA_Vmin])
    idx_fill = forward_interval(i_LV_ES, i_LV_ED, M)
    i_target = i_A if has_A else (idx_fill[int(0.8*(len(idx_fill)-1))] if len(idx_fill)>0 else (i_LA_Vmin+int(0.1*M))%M)
    LA_VpreA = float(y_LA[i_target])

    LA_total_empty = LA_Vmax - LA_Vmin
    LA_reservoir_EF = (LA_total_empty / LA_Vmax * 100.0) if LA_Vmax>0 else np.nan
    LA_passive_EF   = ((LA_Vmax - LA_VpreA) / LA_Vmax * 100.0) if LA_Vmax>0 else np.nan
    LA_active_EF    = ((LA_VpreA - LA_Vmin) / LA_VpreA * 100.0) if LA_VpreA>0 else np.nan

    res.update({
        "LV_EDV": LV_EDV, "LV_ESV": LV_ESV, "LV_SV": LV_SV, "LVEF_%": LVEF,
        "LV_max_eject_vel_ml_perc": LV_max_eject_vel_ml_perc, "t_LV_max_eject_%": t_LV_eject_pct,
        "LV_max_fill_E_vel_ml_perc": LV_E_vel_ml_perc, "t_LV_E_%": t_LV_E_pct,
        "LV_max_fill_A_vel_ml_perc": LV_A_vel_ml_perc, "t_LV_A_%": t_LV_A_pct,
        "LA_Vmax": LA_Vmax, "LA_VpreA": LA_VpreA, "LA_Vmin": LA_Vmin,
        "LA_reservoir_EF_%": LA_reservoir_EF, "LA_passive_EF_%": LA_passive_EF, "LA_active_EF_%": LA_active_EF,
        "LA_LV_coupling_total": (LA_reservoir_EF/LVEF) if (LVEF and not np.isnan(LVEF)) else np.nan,
        "LA_LV_coupling_active": (LA_active_EF/LVEF) if (LVEF and not np.isnan(LVEF)) else np.nan,
        "LA_LV_SV_ratio": (LA_total_empty/LV_SV) if LV_SV!=0 else np.nan
    })
    return res

# ---------- 右心图导出 ----------
def export_patient_plots_and_data(patient_dir: str, metrics: Dict):
    os.makedirs(patient_dir, exist_ok=True)
    t_pct = metrics["_t_pct"]
    y_LA, y_LV, y_RA, y_RV = metrics["_y_LA"], metrics["_y_LV"], metrics["_y_RA"], metrics["_y_RV"]
    d_LA, d_RA, d_LV, d_RV = metrics["_d_LA"], metrics["_d_RA"], metrics["_d_LV"], metrics["_d_RV"]
    idx = metrics["_idx"]

    # 图1：体积曲线
    plt.figure(figsize=(8,5))
    plt.plot(t_pct, y_LA, label="LA")
    plt.plot(t_pct, y_LV, label="LV")
    plt.plot(t_pct, y_RA, label="RA")
    plt.plot(t_pct, y_RV, label="RV")
    for name, i0 in [("ED", idx["RV_ED"]), ("ES", idx["RV_ES"]), ("E", idx["RV_E"])]:
        plt.scatter(t_pct[i0], y_RV[i0], s=30)
        plt.text(t_pct[i0], y_RV[i0], f" RV-{name}", fontsize=9, va="bottom", ha="left")
    if idx["RV_A"] is not None:
        i0 = idx["RV_A"]; plt.scatter(t_pct[i0], y_RV[i0], s=30)
        plt.text(t_pct[i0], y_RV[i0], " RV-A", fontsize=9, va="bottom", ha="left")
    plt.xlabel("% cycle"); plt.ylabel("Volume (ml)")
    plt.title("Volumes (periodic cubic spline + SG)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, "volume_curve.png"), dpi=DPI); plt.close()

    # 图2：dV/dt 曲线（RV/LV）
    plt.figure(figsize=(8,5))
    plt.plot(t_pct, d_RV, label="dV/dt (RV)")
    plt.plot(t_pct, d_LV, label="dV/dt (LV)")
    for tag, i0 in [("RV eject", idx["RV_eject"]), ("LV eject", idx["LV_eject"]),
                    ("RV E", idx["RV_E"]), ("LV E", idx["LV_E"])]:
        val = d_RV[i0] if "RV" in tag else d_LV[i0]
        plt.scatter(t_pct[i0], val, s=30)
        plt.text(t_pct[i0], val, f" {tag}", fontsize=9, va="bottom", ha="left")
    if idx["RV_A"] is not None:
        i0 = idx["RV_A"]; plt.scatter(t_pct[i0], d_RV[i0], s=30)
        plt.text(t_pct[i0], d_RV[i0], " RV A", fontsize=9, va="bottom", ha="left")
    plt.xlabel("% cycle"); plt.ylabel("dV/dt (ml/%cycle)")
    plt.title("dV/dt (periodic cubic spline + SG)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, "dvdt_curve.png"), dpi=DPI); plt.close()

    # 图3：房室耦合：-dRA 与 dRV
    plt.figure(figsize=(8,5))
    plt.plot(t_pct, -d_RA, label="-dV/dt (RA)  ≈ RA emptying")
    plt.plot(t_pct,  d_RV, label=" dV/dt (RV)  ≈ RV filling/ejection")
    plt.scatter(t_pct[idx["RV_E"]],  d_RV[idx["RV_E"]], s=30); plt.text(t_pct[idx["RV_E"]],  d_RV[idx["RV_E"]], " RV E", fontsize=9, va="bottom", ha="left")
    if idx["RV_A"] is not None:
        plt.scatter(t_pct[idx["RV_A"]],  d_RV[idx["RV_A"]], s=30); plt.text(t_pct[idx["RV_A"]],  d_RV[idx["RV_A"]], " RV A", fontsize=9, va="bottom", ha="left")
    plt.xlabel("% cycle"); plt.ylabel("Speed (ml/%cycle)")
    plt.title("RA–RV coupling: -dRA vs dRV")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, "ra_rv_coupling.png"), dpi=DPI); plt.close()

    # 重采样序列表
    curves_df = pd.DataFrame({
        "t_pct": t_pct,
        "LA": y_LA, "LV": y_LV, "RA": y_RA, "RV": y_RV,
        "dVdt_LA_ml_per_pct": d_LA, "dVdt_RA_ml_per_pct": d_RA,
        "dVdt_LV_ml_per_pct": d_LV, "dVdt_RV_ml_per_pct": d_RV
    })
    curves_df.to_csv(os.path.join(patient_dir, "resampled_curves.csv"),
                     index=False, encoding="utf-8-sig")

# ---------- 主流程 ----------
def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"未找到输入文件：{INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    if df.shape[1] < 4:
        raise ValueError("输入表需≥4列：影像号、心腔(LA/LV/RA/RV)、以及≥2个期相体积列。")

    # 列名与基本结构
    df.columns = [str(c).strip() for c in df.columns]
    id_col, label_col = df.columns[0], df.columns[1]
    phase_cols = df.columns[2:]

    # 每4行一例；影像号取组内第一个非空值（兼容“只在第4行标注/只在第1行标注”）
    records = []
    nrows = df.shape[0]
    if nrows % 4 != 0:
        print(f"警告：总行数 {nrows} 不是 4 的倍数，尾部不完整组将被忽略。")
    for i in range(0, nrows, 4):
        block = df.iloc[i:i+4].copy()
        if block.shape[0] < 4: continue
        pid_series = block[id_col].dropna()
        raw_id = pid_series.iloc[0] if len(pid_series) else f"UNK_{i//4+1}"
        pid_val = canonical_id(raw_id)
        block[id_col] = pid_val
        records.append(block)

    df2 = pd.concat(records, ignore_index=True)
    df2[label_col] = df2[label_col].astype(str).str.upper().str.strip()
    df2 = df2[df2[label_col].isin({"LA","LV","RA","RV"})].copy()

    # QC：是否齐全四腔
    qc_counts = df2.groupby(id_col)[label_col].nunique().reset_index(name="n_unique_chambers")
    qc_counts["has_all_four"] = qc_counts["n_unique_chambers"] == 4

    # 逐例计算
    out_dir = os.path.dirname(INPUT_XLSX)
    right_results, left_results = [], []
    for pid, g in df2.groupby(id_col):
        chamber_vols: Dict[str, np.ndarray] = {}
        for lbl, sub in g.groupby(label_col):
            vals = sub[phase_cols].astype(float).values
            vol = vals[0, :] if vals.ndim == 2 else vals
            chamber_vols[lbl] = vol.astype(float)

        # 右心（含绘图用隐藏字段）
        right_metrics = compute_right_metrics(str(pid), chamber_vols, list(phase_cols))
        right_results.append({k: v for k, v in right_metrics.items() if not k.startswith("_")})
        export_patient_plots_and_data(os.path.join(out_dir, safe_dirname(pid)), right_metrics)

        # 左心
        left_metrics = compute_left_metrics(str(pid), chamber_vols, list(phase_cols))
        left_results.append(left_metrics)

    # 汇总 DataFrame
    right_df = pd.DataFrame(right_results)
    left_df  = pd.DataFrame(left_results)

    # 合并 BSA 并指数化
    bsa_df = load_bsa_table(BSA_XLSX)

    right_vol_map = {
        "RV_EDV":"RVEDVi","RV_ESV":"RVESVi","RV_SV":"RVSVi",
        "RA_Vmax":"RAVmaxi","RA_VpreA":"RAVpreAi","RA_Vmin":"RAVmini",
    }
    left_vol_map = {
        "LV_EDV":"LVEDVi","LV_ESV":"LVESVi","LV_SV":"LVSVi",
        "LA_Vmax":"LAVmaxi","LA_VpreA":"LAVpreAi","LA_Vmin":"LAVmini",
    }

    right_idx_df, right_unmatched = add_bsa_indexing_generic(right_df, bsa_df, right_vol_map)
    left_idx_df,  left_unmatched  = add_bsa_indexing_generic(left_df,  bsa_df, left_vol_map)

    # 输出 CSV
    right_df.to_csv(os.path.join(out_dir, "right_heart_metrics_filledIDs.csv"),
                    index=False, encoding="utf-8-sig", float_format="%.6f")
    left_df.to_csv(os.path.join(out_dir, "left_heart_metrics_filledIDs.csv"),
                   index=False, encoding="utf-8-sig", float_format="%.6f")
    if bsa_df is not None:
        right_idx_df.to_csv(os.path.join(out_dir, "right_heart_metrics_indexed.csv"),
                            index=False, encoding="utf-8-sig", float_format="%.6f")
        left_idx_df.to_csv(os.path.join(out_dir, "left_heart_metrics_indexed.csv"),
                           index=False, encoding="utf-8-sig", float_format="%.6f")

    # QC 与未匹配报告
    qc_counts.to_csv(os.path.join(out_dir, "right_heart_metrics_qc.csv"),
                     index=False, encoding="utf-8-sig")
    if bsa_df is not None:
        if not right_unmatched.empty:
            right_unmatched.drop_duplicates().to_csv(os.path.join(out_dir, "bsa_unmatched_ids_right.csv"),
                                                     index=False, encoding="utf-8-sig")
        if not left_unmatched.empty:
            left_unmatched.drop_duplicates().to_csv(os.path.join(out_dir, "bsa_unmatched_ids_left.csv"),
                                                    index=False, encoding="utf-8-sig")

    print(f"完成！右心 {right_df.shape[0]} 例，左心 {left_df.shape[0]} 例。结果路径：{out_dir}")
    if bsa_df is not None:
        print(f"BSA 指数化：右心匹配 {right_idx_df['BSA_m2'].notna().sum()}/{right_idx_df.shape[0]}，"
              f"左心匹配 {left_idx_df['BSA_m2'].notna().sum()}/{left_idx_df.shape[0]}。")

if __name__ == "__main__":
    main()
