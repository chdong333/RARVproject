# -*- coding: utf-8 -*-
"""
右/左心系统功能指标（CT 多期体积 → 拟合曲线）- 改进版V2.5 (修复BSA计算单位错误)

关键修复（相比V2.4）:
- ✅ 修复BSA计算公式：身高必须转换为厘米！
- ✅ 确保指数化值在正常范围内
- ✅ BSA计算正确：√(height_cm × weight_kg / 3600)

V2.4的错误分析:
  V2.4代码使用了错误的BSA公式:
  BSA = √(height(m) × weight(kg) / 3600) ❌
  
  导致:
  - BSA值太小（0.21 instead of 2.08）
  - 指数化值放大10倍（LVEDVi 从60变成600）
  
V2.5修复:
  BSA = √(height(cm) × weight(kg) / 3600) ✅
  
  正确的转换:
  - 身高: 1.73 m → 173 cm
  - 计算: √(173 × 90 / 3600) = 2.08 m²

V2.3继承的功能：
- ✅ 周期样条边界修复
- ✅ 动态E/A窗口调整
- ✅ 多策略VpreA定位
- ✅ 完整的左右心功能计算
- ✅ 统一的QC质量控制系统

输出文件:
- left_heart_metrics_v2.csv
- right_heart_metrics_v2.csv
- left_heart_metrics_indexed_v2.csv ✅ 修复
- right_heart_metrics_indexed_v2.csv ✅ 修复
- quality_control_report_v2.json
- chamber_completeness_qc.csv
"""

import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks

# ======== 路径（按需修改） ========
INPUT_XLSX = r"D:\数据资料\联影电影数据\rarvproj\ex3.XLSX"
BSA_XLSX   = r"D:\数据资料\联影电影数据\rarvproj\BSA3.xlsx"

# ======== 参数 ========
UPSAMPLE_POINTS = 4000
SG_WINDOW_FRAC  = 0.02
SG_POLY         = 3
PEAK_MIN_DISTANCE_PCT = 5.0
PEAK_MIN_PROM_STD     = 0.5
DPI = 150

# ======== QC质量控制类 ========
class CardiacAnalysisQC:
    """质量控制标记系统"""
    
    def __init__(self, patient_id=None):
        self.patient_id = patient_id
        self.flags = {}
        self.warnings = []
        self.risk_scores = {}
        self.notes = {}
    
    def add_flag(self, category: str, key: str, value):
        if category not in self.flags:
            self.flags[category] = {}
        self.flags[category][key] = value
    
    def add_warning(self, level: str, message: str):
        self.warnings.append({'level': level, 'message': message})
    
    def add_note(self, key: str, message: str):
        self.notes[key] = message
    
    def to_dict(self):
        return {
            'flags': self.flags,
            'warnings': self.warnings,
            'risk_scores': self.risk_scores,
            'notes': self.notes
        }
    
    def to_csv_row(self):
        return {
            'QC_flags_count': sum(len(v) for v in self.flags.values()),
            'QC_warnings_count': len(self.warnings),
            'QC_warnings': '; '.join([f"{w['level']}: {w['message']}" for w in self.warnings]),
            'QC_risk_level': self.risk_scores.get('overall', 'UNKNOWN')
        }

# ---------- 基础工具 ----------
def safe_dirname(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s).replace(" ", "_")
    return s if s else "UNK"

def canonical_id(val) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)): 
        return ""
    try:
        if isinstance(val, (int, np.integer)): 
            return str(int(val))
        if isinstance(val, (float, np.floating)):
            if np.isfinite(val) and float(val).is_integer(): 
                return str(int(val))
            s = format(val, "f").rstrip("0").rstrip(".")
        else:
            s = str(val)
    except Exception:
        s = str(val)
    s = s.strip()
    if s.endswith(".0"): 
        s = s[:-2]
    return s.replace(" ", "")

def _savgol(y: np.ndarray, win_frac: float, poly: int) -> np.ndarray:
    n = len(y)
    w = max(7, int(n * win_frac))
    if w % 2 == 0: 
        w += 1
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if w < poly + 2:
        w = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if w < 7: 
        return y
    return savgol_filter(y, window_length=w, polyorder=poly, mode="wrap")

def resample_with_periodic_spline(vols: np.ndarray,
                                  upsample_M: int = UPSAMPLE_POINTS,
                                  sg_frac: float = SG_WINDOW_FRAC,
                                  sg_poly: int = SG_POLY) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CardiacAnalysisQC]:
    """改进版周期样条重采样（带周期边界修复）"""
    qc = CardiacAnalysisQC()
    vols = np.asarray(vols, dtype=float).copy()
    N = vols.shape[0]
    t_orig = np.linspace(0.0, 1.0, N, endpoint=False)

    # 填补NaN/inf
    idx = np.arange(N)
    mask = np.isfinite(vols)
    if not mask.all():
        if not mask.any():
            qc.add_warning('CRITICAL', '所有体积值都无效')
            raise ValueError("所有体积值都无效")
        vols[~mask] = np.interp(idx[~mask], idx[mask], vols[mask])
        qc.add_flag('resampling', 'NaN_filled', (~mask).sum())

    y_smooth_coarse = _savgol(vols, sg_frac * 1.5, sg_poly)
    ed_idx_initial = int(np.argmax(y_smooth_coarse))
    es_idx_initial = int(np.argmin(y_smooth_coarse))
    
    ed_val = vols[ed_idx_initial]
    es_val = vols[es_idx_initial]
    mean_vol = np.mean(vols)
    std_vol = np.std(vols)
    
    qc.add_flag('ED_ES', 'ED_index', ed_idx_initial)
    qc.add_flag('ED_ES', 'ES_index', es_idx_initial)
    qc.add_flag('ED_ES', 'ED_value', float(ed_val))
    qc.add_flag('ED_ES', 'ES_value', float(es_val))

    if ed_val > mean_vol + 4 * std_vol or es_val < mean_vol - 3 * std_vol:
        qc.add_warning('WARNING', f'体积振幅异常：ED={ed_val:.1f}, ES={es_val:.1f}')
        qc.add_flag('ED_ES', 'amplitude_warning', True)

    # 修复：严格的周期边界检查
    first_val = vols[0]
    last_val = vols[-1]
    is_periodic = np.isclose(first_val, last_val, rtol=1e-10, atol=1e-12)
    
    if is_periodic:
        t_fit = t_orig
        y_fit = vols.copy()
        bc_type = "periodic"
        qc.add_flag('interpolation', 'boundary_type', 'periodic')
        qc.add_flag('interpolation', 'boundary_diff', float(abs(first_val - last_val)))
    else:
        t_fit = np.concatenate([[0.0], t_orig, [1.0]])
        closure_val = (first_val + last_val) / 2.0
        y_fit = np.concatenate([[closure_val], vols, [closure_val]])
        bc_type = "periodic"
        qc.add_flag('interpolation', 'boundary_type', 'periodic_forced')
        qc.add_flag('interpolation', 'boundary_diff', float(abs(first_val - last_val)))
        qc.add_flag('interpolation', 'closure_applied', True)
        qc.add_warning('INFO', f'首尾不等({abs(first_val-last_val):.3f}mL)，已强制闭合')

    try:
        cs = CubicSpline(t_fit, y_fit, bc_type=bc_type)
    except ValueError as e:
        qc.add_warning('WARNING', f'周期样条失败，回退到natural边界: {str(e)[:50]}')
        t_fit_backup = np.concatenate([[-1/N], t_orig, [1 + 1/N]])
        y_fit_backup = np.concatenate([[vols[-1]], vols, [vols[0]]])
        cs = CubicSpline(t_fit_backup, y_fit_backup, bc_type="natural")
        qc.add_flag('interpolation', 'boundary_type', 'natural_fallback')

    t_hr = np.linspace(0.0, 1.0, upsample_M, endpoint=False)
    y_hr = cs(t_hr)
    dy_hr = cs(t_hr, 1)

    y_sm = _savgol(y_hr, sg_frac, sg_poly)
    dy_sm = _savgol(dy_hr, sg_frac, sg_poly)
    
    dy_abs_max = np.max(np.abs(dy_sm))
    dy_mean = np.mean(np.abs(dy_sm))
    
    if dy_mean > 0:
        dy_ratio = dy_abs_max / dy_mean
        qc.add_flag('derivative', 'max_ratio_to_mean', float(dy_ratio))
        
        if dy_ratio > 10:
            qc.add_warning('WARNING', f'导数振荡异常：max/mean={dy_ratio:.2f}')
            dy_sm = _savgol(dy_sm, max(sg_frac * 2, 0.05), sg_poly)
            qc.add_flag('derivative', 'extra_smoothing_applied', True)

    buffer = int(upsample_M * 0.02)
    y_sm[:buffer] = y_sm[buffer]
    y_sm[-buffer:] = y_sm[-buffer-1]
    dy_sm[:buffer] = dy_sm[buffer]
    dy_sm[-buffer:] = dy_sm[-buffer-1]
    qc.add_flag('boundary', 'buffer_size', buffer)

    return t_hr, y_sm, dy_sm, qc

def forward_interval(i_start: int, i_end: int, M: int) -> np.ndarray:
    return np.arange(i_start, i_end + 1) if i_start <= i_end else \
           np.concatenate([np.arange(i_start, M), np.arange(0, i_end + 1)])

def _progress_in_arc(idx: int, start: int, end: int, M: int) -> float:
    if start <= end:
        length = end - start + 1
        pos = idx - start
    else:
        length = (M - start) + (end + 1)
        pos = (idx - start) if idx >= start else (M - start) + idx
    return max(0.0, min(1.0, pos / max(1, length - 1)))

def find_EA_windowed_v2(drv: np.ndarray, idx_ES: int, idx_ED: int, M: int,
                        e_win_default=(0.05, 0.40), 
                        a_win_default=(0.45, 0.95),
                        min_dist_pct=PEAK_MIN_DISTANCE_PCT, 
                        prom_std=PEAK_MIN_PROM_STD, 
                        edge_buffer=0.01) -> Tuple[Optional[int], Optional[int], CardiacAnalysisQC]:
    """改进的E/A峰值查找"""
    qc = CardiacAnalysisQC()
    seg = forward_interval(idx_ES, idx_ED, M)
    min_distance = max(3, int(M * (min_dist_pct / 100.0)))
    prom = max(1e-6, float(np.nanstd(drv)) * prom_std)
    
    try:
        peaks, props = find_peaks(drv, distance=min_distance, prominence=prom)
    except:
        qc.add_warning('WARNING', 'find_peaks失败')
        return None, None, qc

    seg_set = set(seg)
    peaks_in = []
    for p in peaks:
        if p in seg_set:
            pr = _progress_in_arc(p, idx_ES, idx_ED, M)
            if edge_buffer <= pr <= 1.0 - edge_buffer:
                try:
                    prom_val = props['prominences'][peaks.tolist().index(p)]
                    peaks_in.append((p, pr, prom_val))
                except:
                    peaks_in.append((p, pr, prom))
    
    qc.add_flag('EA_detection', 'peaks_found', len(peaks_in))
    
    if not peaks_in:
        max_idx = seg[int(np.argmax(drv[seg]))]
        qc.add_flag('EA_detection', 'strategy', 'no_peaks_fallback')
        return max_idx, None, qc
    
    peaks_in.sort(key=lambda x: x[1])
    
    lo_e, hi_e = e_win_default
    lo_a, hi_a = a_win_default
    
    e_cands = [(p, pr, prom) for p, pr, prom in peaks_in if lo_e <= pr <= hi_e]
    a_cands = [(p, pr, prom) for p, pr, prom in peaks_in if lo_a <= pr <= hi_a]
    
    if e_cands and a_cands:
        iE = e_cands[0][0]
        iA = a_cands[-1][0]
        qc.add_flag('EA_detection', 'strategy', 'separated')
        return iE, iA, qc
    
    if len(peaks_in) >= 2:
        peaks_sorted = sorted(peaks_in, key=lambda x: x[2], reverse=True)
        candidates = sorted([peaks_sorted[0][:2], peaks_sorted[1][:2]], 
                          key=lambda x: x[1])
        qc.add_flag('EA_detection', 'strategy', 'fusion_detected')
        return candidates[0][0], candidates[1][0], qc
    
    if len(peaks_in) == 1:
        qc.add_flag('EA_detection', 'strategy', 'single_peak_only')
        qc.add_warning('INFO', 'A波缺失：充盈期仅有单个峰值')
        return peaks_in[0][0], None, qc
    
    qc.add_warning('WARNING', '无法识别E/A波')
    return None, None, qc

def find_eject_windowed_v2(drv: np.ndarray, idx_ED: int, idx_ES: int, M: int,
                           ej_win=(0.05, 0.45)) -> Tuple[Optional[int], CardiacAnalysisQC]:
    """改进的射出速率峰值查找"""
    qc = CardiacAnalysisQC()
    seg = forward_interval(idx_ED, idx_ES, M)
    
    lo_frac, hi_frac = ej_win
    seg_len = len(seg)
    lo_idx = int(seg_len * lo_frac)
    hi_idx = int(seg_len * hi_frac)
    ej_seg = seg[lo_idx:hi_idx] if lo_idx < hi_idx else seg
    
    if len(ej_seg) == 0:
        ej_seg = seg
    
    best_idx = ej_seg[int(np.argmin(drv[ej_seg]))]
    best_val = drv[best_idx]
    mean_drv = np.mean(drv)
    std_drv = np.std(drv)
    
    qc.add_flag('eject_detection', 'peak_value', float(best_val))
    qc.add_flag('eject_detection', 'window_size', len(ej_seg))
    
    if std_drv > 0:
        z_score = (best_val - mean_drv) / std_drv
        qc.add_flag('eject_detection', 'z_score', float(z_score))
        
        if z_score < -5:
            sorted_idxs = np.argsort(drv[ej_seg])
            if len(sorted_idxs) > 1:
                best_idx = ej_seg[sorted_idxs[1]]
                qc.add_warning('INFO', f'射出峰值过极端，使用次优值')
                qc.add_flag('eject_detection', 'fallback_used', True)
    
    return best_idx, qc

def compute_cardiac_metrics_v2(patient_id: str,
                               chamber_vols: Dict[str, np.ndarray],
                               phase_names: List[str],
                               chamber_type: str = "left") -> Tuple[Dict, CardiacAnalysisQC]:
    """统一的左右心功能指标计算"""
    qc_main = CardiacAnalysisQC(patient_id)
    chamber_pair = {"left": ("LA", "LV"), "right": ("RA", "RV")}[chamber_type]
    atrium_name, ventricle_name = chamber_pair
    
    res = {
        "PatientID": canonical_id(patient_id), 
        "N_phases": len(phase_names)
    }
    
    output_fields = [
        f"{ventricle_name}_EDV", f"{ventricle_name}_ESV", f"{ventricle_name}_SV", f"{ventricle_name}EF_%",
        f"{ventricle_name}_max_eject_vel_ml_perc", f"t_{ventricle_name}_max_eject_%",
        f"{ventricle_name}_max_fill_E_vel_ml_perc", f"t_{ventricle_name}_E_%",
        f"{ventricle_name}_max_fill_A_vel_ml_perc", f"t_{ventricle_name}_A_%",
        f"{atrium_name}_Vmax", f"{atrium_name}_VpreA", f"{atrium_name}_Vmin",
        f"{atrium_name}_reservoir_EF_%", f"{atrium_name}_passive_EF_%", f"{atrium_name}_active_EF_%",
        f"{atrium_name}_{ventricle_name}_coupling_total", f"{atrium_name}_{ventricle_name}_coupling_active", 
        f"{atrium_name}_{ventricle_name}_SV_ratio"
    ]
    
    need = {atrium_name, ventricle_name}
    if not need.issubset(chamber_vols.keys()):
        qc_main.add_warning('CRITICAL', f'缺少必要心腔数据: {need}')
        for k in output_fields:
            res[k] = np.nan
        return res, qc_main

    t_at, y_at, d_at, qc_at = resample_with_periodic_spline(chamber_vols[atrium_name])
    t_vt, y_vt, d_vt, qc_vt = resample_with_periodic_spline(chamber_vols[ventricle_name])
    M = len(t_vt)
    
    qc_main.add_flag('resampling', atrium_name, qc_at.to_dict())
    qc_main.add_flag('resampling', ventricle_name, qc_vt.to_dict())

    i_vt_ED = int(np.argmax(y_vt))
    i_vt_ES = int(np.argmin(y_vt))
    vt_EDV, vt_ESV = float(y_vt[i_vt_ED]), float(y_vt[i_vt_ES])
    vt_SV = vt_EDV - vt_ESV
    vt_EF = (vt_SV / vt_EDV * 100.0) if vt_EDV > 0 else np.nan
    
    if vt_EDV > 250 or vt_ESV > 100:
        qc_main.add_warning('WARNING', f'{ventricle_name}体积极端：EDV={vt_EDV:.1f}, ESV={vt_ESV:.1f}')
        qc_main.add_flag('ED_ES', 'extreme_volume_warning', True)

    i_vt_eject, qc_eject = find_eject_windowed_v2(d_vt, i_vt_ED, i_vt_ES, M)
    if i_vt_eject is not None:
        vt_max_eject_vel = float(d_vt[i_vt_eject]) / 100.0
        t_vt_eject_pct = i_vt_eject / M * 100.0
    else:
        vt_max_eject_vel = np.nan
        t_vt_eject_pct = np.nan
    
    if not np.isnan(t_vt_eject_pct) and (t_vt_eject_pct < 5 or t_vt_eject_pct > 30):
        qc_main.add_warning('WARNING', f'{ventricle_name}射出时间异常：{t_vt_eject_pct:.1f}%')
        qc_main.add_flag('eject_detection', 'time_abnormal', float(t_vt_eject_pct))
    
    qc_main.add_flag('eject_detection', ventricle_name, qc_eject.to_dict())

    i_E, i_A, qc_ea = find_EA_windowed_v2(d_vt, i_vt_ES, i_vt_ED, M,
                                           e_win_default=(0.05, 0.45),
                                           a_win_default=(0.45, 0.95))
    qc_main.add_flag('EA_detection', ventricle_name, qc_ea.to_dict())
    
    if i_E is not None:
        vt_E_vel = float(d_vt[i_E]) / 100.0
        t_vt_E_pct = i_E / M * 100.0
    else:
        vt_E_vel = np.nan
        t_vt_E_pct = np.nan
    
    if i_A is not None:
        vt_A_vel = float(d_vt[i_A]) / 100.0
        t_vt_A_pct = i_A / M * 100.0
        has_A = True
    else:
        vt_A_vel = np.nan
        t_vt_A_pct = np.nan
        has_A = False
        qc_main.add_warning('INFO', f'{ventricle_name}A波缺失')

    i_at_Vmax = int(np.argmax(y_at))
    i_at_Vmin = int(np.argmin(y_at))
    at_Vmax = float(y_at[i_at_Vmax])
    at_Vmin = float(y_at[i_at_Vmin])
    idx_fill = forward_interval(i_vt_ES, i_vt_ED, M)
    
    if has_A and i_A is not None:
        at_VpreA = float(y_at[i_A])
        vpreA_strategy = "A_wave_position"
    else:
        if len(idx_fill) > 10:
            at_fill_vals = y_at[idx_fill]
            at_fill_mins = np.argsort(at_fill_vals)[:3]
            
            for min_idx in at_fill_mins:
                candidate_i = idx_fill[min_idx]
                if candidate_i != i_at_Vmin:
                    at_VpreA = float(y_at[candidate_i])
                    vpreA_strategy = "local_minimum_in_fill"
                    break
            else:
                at_VpreA = float(y_at[idx_fill[int(0.5 * (len(idx_fill)-1))]])
                vpreA_strategy = "midpoint_fill"
        else:
            at_VpreA = float(y_at[idx_fill[int(0.5 * (len(idx_fill)-1))]]) if len(idx_fill)>0 else at_Vmin
            vpreA_strategy = "fallback_midpoint"
    
    if at_VpreA <= at_Vmin * 1.05:
        qc_main.add_warning('WARNING', f'{atrium_name}_VpreA异常接近{atrium_name}_Vmin，执行修正')
        at_VpreA = at_Vmin + 0.3 * (at_Vmax - at_Vmin)
        vpreA_strategy += "_corrected"
        qc_main.add_flag(f'{atrium_name}_VpreA', 'correction_applied', True)
    
    qc_main.add_flag(f'{atrium_name}_VpreA', 'strategy_used', vpreA_strategy)

    at_total_empty = at_Vmax - at_Vmin
    at_reservoir_EF = (at_total_empty / at_Vmax * 100.0) if at_Vmax > 0 else np.nan
    at_passive_EF = ((at_Vmax - at_VpreA) / at_Vmax * 100.0) if at_Vmax > 0 else np.nan
    at_active_EF = ((at_VpreA - at_Vmin) / at_VpreA * 100.0) if at_VpreA > 0 else np.nan
    
    if not np.isnan(at_active_EF):
        if at_active_EF < 15:
            qc_main.add_warning('WARNING', f'{atrium_name}_active_EF极端低: {at_active_EF:.2f}%')
            qc_main.add_flag('risk_score', f'{atrium_name}_afib_risk_level', 'HIGH')
        else:
            qc_main.add_flag('risk_score', f'{atrium_name}_afib_risk_level', 'NORMAL')

    if not np.isnan(vt_E_vel):
        if vt_E_vel < 1.5:
            qc_main.add_warning('WARNING', f'{ventricle_name}E波速度极低: {vt_E_vel:.4f}，严重舒张障碍')
            qc_main.add_flag('risk_score', 'diastolic_function', 'SEVERE_IMPAIRMENT')
        elif vt_E_vel < 2.5:
            qc_main.add_flag('risk_score', 'diastolic_function', 'MODERATE_IMPAIRMENT')
    
    if not np.isnan(t_vt_E_pct) and (t_vt_E_pct < 20 or t_vt_E_pct > 50):
        qc_main.add_warning('INFO', f'{ventricle_name}E波时间延迟: {t_vt_E_pct:.1f}%')

    res.update({
        f"{ventricle_name}_EDV": vt_EDV,
        f"{ventricle_name}_ESV": vt_ESV,
        f"{ventricle_name}_SV": vt_SV,
        f"{ventricle_name}EF_%": vt_EF,
        f"{ventricle_name}_max_eject_vel_ml_perc": vt_max_eject_vel,
        f"t_{ventricle_name}_max_eject_%": t_vt_eject_pct,
        f"{ventricle_name}_max_fill_E_vel_ml_perc": vt_E_vel,
        f"t_{ventricle_name}_E_%": t_vt_E_pct,
        f"{ventricle_name}_max_fill_A_vel_ml_perc": vt_A_vel,
        f"t_{ventricle_name}_A_%": t_vt_A_pct,
        f"{atrium_name}_Vmax": at_Vmax,
        f"{atrium_name}_VpreA": at_VpreA,
        f"{atrium_name}_Vmin": at_Vmin,
        f"{atrium_name}_reservoir_EF_%": at_reservoir_EF,
        f"{atrium_name}_passive_EF_%": at_passive_EF,
        f"{atrium_name}_active_EF_%": at_active_EF,
        f"{atrium_name}_{ventricle_name}_coupling_total": (at_reservoir_EF/vt_EF) if (vt_EF and not np.isnan(vt_EF)) else np.nan,
        f"{atrium_name}_{ventricle_name}_coupling_active": (at_active_EF/vt_EF) if (vt_EF and not np.isnan(vt_EF)) else np.nan,
        f"{atrium_name}_{ventricle_name}_SV_ratio": (at_total_empty/vt_SV) if vt_SV != 0 else np.nan,
        "_t_pct": np.linspace(0, 100, M, endpoint=False),
        f"_y_{atrium_name}": y_at,
        f"_y_{ventricle_name}": y_vt,
        f"_d_{atrium_name}": d_at/100.0,
        f"_d_{ventricle_name}": d_vt/100.0,
        "_idx": {f"{ventricle_name}_ED": i_vt_ED, f"{ventricle_name}_ES": i_vt_ES, 
                f"{ventricle_name}_E": i_E, f"{ventricle_name}_A": i_A, f"{ventricle_name}_eject": i_vt_eject}
    })
    
    return res, qc_main

def export_patient_plots_and_data(patient_dir: str, metrics: Dict, chamber_type: str = "left"):
    """导出图表和数据"""
    os.makedirs(patient_dir, exist_ok=True)
    
    pair = ("LA", "LV") if chamber_type == "left" else ("RA", "RV")
    at_name, vt_name = pair
    
    t_pct = metrics["_t_pct"]
    y_at = metrics[f"_y_{at_name}"]
    y_vt = metrics[f"_y_{vt_name}"]
    d_at = metrics[f"_d_{at_name}"]
    d_vt = metrics[f"_d_{vt_name}"]
    idx = metrics["_idx"]

    plt.figure(figsize=(8,5))
    plt.plot(t_pct, y_at, label=at_name)
    plt.plot(t_pct, y_vt, label=vt_name)
    for name, i0 in [(f"{vt_name}_ED", idx[f"{vt_name}_ED"]), 
                     (f"{vt_name}_ES", idx[f"{vt_name}_ES"]), 
                     (f"{vt_name}_E", idx[f"{vt_name}_E"])]:
        if i0 is not None:
            plt.scatter(t_pct[i0], y_vt[i0], s=30)
            plt.text(t_pct[i0], y_vt[i0], f" {name}", fontsize=9, va="bottom", ha="left")
    plt.xlabel("% cycle")
    plt.ylabel("Volume (ml)")
    plt.title(f"{chamber_type.upper()} Heart Volumes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, f"{chamber_type}_volume_curve.png"), dpi=DPI)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(t_pct, d_vt, label=f"dV/dt ({vt_name})")
    plt.plot(t_pct, d_at, label=f"dV/dt ({at_name})")
    for tag, i0 in [(f"{vt_name}_eject", idx[f"{vt_name}_eject"]), 
                    (f"{vt_name}_E", idx[f"{vt_name}_E"])]:
        if i0 is not None:
            plt.scatter(t_pct[i0], d_vt[i0], s=30)
            plt.text(t_pct[i0], d_vt[i0], f" {tag}", fontsize=9, va="bottom", ha="left")
    plt.xlabel("% cycle")
    plt.ylabel("dV/dt (ml/%cycle)")
    plt.title(f"{chamber_type.upper()} Heart dV/dt Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dir, f"{chamber_type}_dvdt_curve.png"), dpi=DPI)
    plt.close()

    curves_df = pd.DataFrame({
        "t_pct": t_pct,
        at_name: y_at,
        vt_name: y_vt,
        f"dVdt_{at_name}_ml_per_pct": d_at,
        f"dVdt_{vt_name}_ml_per_pct": d_vt
    })
    curves_df.to_csv(os.path.join(patient_dir, f"{chamber_type}_resampled_curves.csv"),
                     index=False, encoding="utf-8-sig")

def load_bsa_table(bsa_path: str) -> Optional[pd.DataFrame]:
    """
    加载BSA表并计算BSA值 - V2.5修复版
    
    支持的列名:
    - 患者ID: 影像号, PatientID, CaseID, ID, 编号 等
    - 身高: high, 身高, height, height_cm (单位: 米 m)
    - 体重: weight, 体重 (单位: kg)
    
    BSA计算公式 (DuBois/Mosteller): 
    BSA(m²) = √(身高(cm) × 体重(kg) / 3600)
    
    V2.5修复: 必须将身高从米转换为厘米！
    """
    if not bsa_path or not os.path.exists(bsa_path):
        print("  ⚠ 未找到BSA文件，跳过指数化")
        return None
    
    try:
        df = pd.read_excel(bsa_path, engine="openpyxl")
        print(f"  ✓ 读取BSA文件成功，包含 {len(df)} 行数据")
    except Exception as e:
        print(f"  ✗ 读取BSA文件失败: {e}")
        return None

    print(f"  列名: {list(df.columns)}")

    # 1. 查找患者ID列
    id_col = None
    for c in df.columns:
        cstr = str(c).strip().lower()
        if cstr in ["影像号", "patientid", "caseid", "id", "编号"] or "影像" in cstr:
            id_col = c
            print(f"  ✓ 找到患者ID列: {c}")
            break
    
    if id_col is None:
        print(f"  ✗ 未找到患者ID列，可用列: {list(df.columns)}")
        return None

    # 2. 查找身高列（米）
    height_col = None
    for c in df.columns:
        cstr = str(c).strip().lower()
        if cstr in ["high", "height", "身高", "height_cm"]:
            height_col = c
            print(f"  ✓ 找到身高列: {c}")
            break
    
    # 3. 查找体重列（kg）
    weight_col = None
    for c in df.columns:
        cstr = str(c).strip().lower()
        if cstr in ["weight", "体重", "weight_kg"]:
            weight_col = c
            print(f"  ✓ 找到体重列: {c}")
            break
    
    if height_col is None or weight_col is None:
        print(f"  ✗ 缺少身高或体重列")
        print(f"     可用列: {list(df.columns)}")
        return None

    # 3. 构建输出
    out = pd.DataFrame()
    out["PatientID"] = df[id_col].apply(canonical_id)
    
    # 4. 提取身高和体重
    height = pd.to_numeric(df[height_col], errors="coerce")
    weight = pd.to_numeric(df[weight_col], errors="coerce")
    
    # 检查身高单位并转换为厘米
    height_median = height.dropna().median()
    if height_median > 3.0:
        print(f"  ⚠ 身高中位数 {height_median:.2f} > 3，认为已为厘米，不再转换")
        height_cm = height
    elif height_median < 0.5:
        print(f"  ✗ 身高中位数 {height_median:.4f} < 0.5，数据异常")
        return None
    else:
        print(f"  ✓ 身高中位数 {height_median:.2f}m，需要转换为cm")
        height_cm = height * 100  # ✅ 修复：身高从米转为厘米
    
    # 检查体重单位
    weight_median = weight.dropna().median()
    if weight_median > 300:
        print(f"  ⚠ 体重中位数 {weight_median:.1f} > 300，认为是克，转换为kg")
        weight = weight / 1000.0
    elif weight_median < 10:
        print(f"  ✗ 体重中位数 {weight_median:.1f} < 10，数据异常")
        return None
    else:
        print(f"  ✓ 体重中位数 {weight_median:.1f}kg，单位正确")
    
    # 5. 计算BSA（修复版本 - 身高必须是cm！）
    # 公式: BSA(m²) = √(height(cm) × weight(kg) / 3600)
    print(f"\n  【BSA计算】")
    print(f"  公式: BSA = √(身高(cm) × 体重(kg) / 3600)")
    out["BSA_m2"] = np.sqrt(height_cm * weight / 3600.0)
    
    # 移除无效行
    out = out.replace([np.inf, -np.inf], np.nan)
    out_valid = out.dropna(subset=["BSA_m2"])
    
    print(f"  成功计算 {len(out_valid)}/{len(out)} 例BSA值")
    
    # 验证BSA值范围
    bsa_min = out_valid["BSA_m2"].min()
    bsa_max = out_valid["BSA_m2"].max()
    bsa_mean = out_valid["BSA_m2"].mean()
    
    if bsa_min < 1.0 or bsa_max > 3.0 or bsa_mean < 1.5:
        print(f"  ⚠ BSA值范围: {bsa_min:.2f} - {bsa_max:.2f} m² (平均 {bsa_mean:.2f})")
        print(f"  ⚠ 可能仍有问题！正常范围应为 1.2-2.5 m²")
    else:
        print(f"  ✓ BSA值范围: {bsa_min:.2f} - {bsa_max:.2f} m² (平均 {bsa_mean:.2f})")
    
    # 去重
    out_dedup = out_valid.groupby("PatientID", as_index=False)["BSA_m2"].first()
    
    return out_dedup

def add_bsa_indexing_generic(res_df: pd.DataFrame, bsa_df: Optional[pd.DataFrame],
                             vol_map: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    合并BSA并指数化
    
    返回: (指数化后的df, 未匹配的患者列表)
    """
    if bsa_df is None or bsa_df.empty:
        print(f"  ⚠ BSA数据为空，无法指数化")
        return res_df.copy(), []
    
    df = res_df.copy()
    df["PatientID"] = df["PatientID"].apply(canonical_id)
    bsa_df = bsa_df.copy()
    bsa_df["PatientID"] = bsa_df["PatientID"].apply(canonical_id)
    
    # 合并
    merged = df.merge(bsa_df, on="PatientID", how="left", suffixes=("", "_bsa"))
    if "BSA_m2_bsa" in merged.columns:
        merged["BSA_m2"] = merged["BSA_m2_bsa"]
        merged.drop(columns=["BSA_m2_bsa"], inplace=True)
    
    # 统计未匹配患者
    unmatched = merged[merged["BSA_m2"].isna()]["PatientID"].tolist()
    if unmatched:
        print(f"  ⚠ {len(unmatched)}例患者未找到BSA数据")

    # 指数化函数
    def idx_div(x, bsa):
        try:
            return x / bsa if (bsa is not None and np.isfinite(bsa) and bsa > 0) else np.nan
        except:
            return np.nan

    # 执行指数化
    for src, dst in vol_map.items():
        if src in merged.columns:
            merged[dst] = [idx_div(v, b) for v, b in zip(merged[src].values, merged["BSA_m2"].values)]
    
    return merged, unmatched

# ========== 主程序 ==========
def main():
    print("=" * 100)
    print("心脏功能分析 V2.5 - 完整左右心输出（修复BSA计算单位）")
    print("=" * 100)
    
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"未找到文件：{INPUT_XLSX}")
    
    print(f"\n读取输入文件: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    if df.shape[1] < 4:
        raise ValueError("输入表需≥4列")

    df.columns = [str(c).strip() for c in df.columns]
    id_col, label_col = df.columns[0], df.columns[1]
    phase_cols = df.columns[2:]

    # 数据预处理
    records = []
    nrows = df.shape[0]
    if nrows % 4 != 0:
        print(f"警告：行数{nrows}不是4的倍数")
    
    for i in range(0, nrows, 4):
        block = df.iloc[i:i+4].copy()
        if block.shape[0] < 4:
            continue
        pid_series = block[id_col].dropna()
        raw_id = pid_series.iloc[0] if len(pid_series) else f"UNK_{i//4+1}"
        pid_val = canonical_id(raw_id)
        block[id_col] = pid_val
        records.append(block)

    df2 = pd.concat(records, ignore_index=True)
    df2[label_col] = df2[label_col].astype(str).str.upper().str.strip()
    df2 = df2[df2[label_col].isin({"LA", "LV", "RA", "RV"})].copy()

    qc_counts = df2.groupby(id_col)[label_col].nunique().reset_index(name="n_unique_chambers")
    qc_counts["has_all_four"] = qc_counts["n_unique_chambers"] == 4

    out_dir = os.path.dirname(INPUT_XLSX)
    left_results = []
    right_results = []
    all_qc_records = []
    
    print(f"\n处理患者数据...")
    for pid, g in df2.groupby(id_col):
        chamber_vols: Dict[str, np.ndarray] = {}
        for lbl, sub in g.groupby(label_col):
            vals = sub[phase_cols].astype(float).values
            vol = vals[0, :] if vals.ndim == 2 else vals
            chamber_vols[lbl] = vol.astype(float)

        # 左心
        if {"LA", "LV"}.issubset(chamber_vols.keys()):
            left_metrics, qc_left = compute_cardiac_metrics_v2(str(pid), chamber_vols, list(phase_cols), chamber_type="left")
            left_metrics.update(qc_left.to_csv_row())
            left_results.append({k: v for k, v in left_metrics.items() if not k.startswith("_")})
            all_qc_records.append({"PatientID": pid, "chamber": "LEFT", **qc_left.to_dict()})
            
            if all(k in left_metrics for k in ["_y_LV", "_y_LA"]):
                export_patient_plots_and_data(os.path.join(out_dir, safe_dirname(pid)), left_metrics, "left")
        
        # 右心
        if {"RA", "RV"}.issubset(chamber_vols.keys()):
            right_metrics, qc_right = compute_cardiac_metrics_v2(str(pid), chamber_vols, list(phase_cols), chamber_type="right")
            right_metrics.update(qc_right.to_csv_row())
            right_results.append({k: v for k, v in right_metrics.items() if not k.startswith("_")})
            all_qc_records.append({"PatientID": pid, "chamber": "RIGHT", **qc_right.to_dict()})
            
            if all(k in right_metrics for k in ["_y_RV", "_y_RA"]):
                export_patient_plots_and_data(os.path.join(out_dir, safe_dirname(pid)), right_metrics, "right")

    # 生成输出
    print(f"\n生成输出文件...")
    left_df = pd.DataFrame(left_results)
    right_df = pd.DataFrame(right_results)
    
    print(f"\n加载BSA文件...")
    bsa_df = load_bsa_table(BSA_XLSX)

    # 左心
    left_df.to_csv(os.path.join(out_dir, "left_heart_metrics_v2.csv"),
                   index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"  ✓ left_heart_metrics_v2.csv ({len(left_df)}行)")
    
    # 右心
    right_df.to_csv(os.path.join(out_dir, "right_heart_metrics_v2.csv"),
                    index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"  ✓ right_heart_metrics_v2.csv ({len(right_df)}行)")
    
    # 指数化
    if bsa_df is not None and not bsa_df.empty:
        print(f"\n生成指数化文件...")
        
        # 左心指数化
        left_vol_map = {
            "LV_EDV": "LVEDVi", "LV_ESV": "LVESVi", "LV_SV": "LVSVi",
            "LA_Vmax": "LAVmaxi", "LA_VpreA": "LAVpreAi", "LA_Vmin": "LAVmini",
        }
        left_idx_df, left_unmatched = add_bsa_indexing_generic(left_df, bsa_df, left_vol_map)
        left_idx_df.to_csv(os.path.join(out_dir, "left_heart_metrics_indexed_v2.csv"),
                           index=False, encoding="utf-8-sig", float_format="%.6f")
        print(f"  ✓ left_heart_metrics_indexed_v2.csv ({len(left_idx_df)}行，{left_idx_df['BSA_m2'].notna().sum()}例有BSA值)")
        
        # 右心指数化
        right_vol_map = {
            "RV_EDV": "RVEDVi", "RV_ESV": "RVESVi", "RV_SV": "RVSVi",
            "RA_Vmax": "RAVmaxi", "RA_VpreA": "RAVpreAi", "RA_Vmin": "RAVmini",
        }
        right_idx_df, right_unmatched = add_bsa_indexing_generic(right_df, bsa_df, right_vol_map)
        right_idx_df.to_csv(os.path.join(out_dir, "right_heart_metrics_indexed_v2.csv"),
                            index=False, encoding="utf-8-sig", float_format="%.6f")
        print(f"  ✓ right_heart_metrics_indexed_v2.csv ({len(right_idx_df)}行，{right_idx_df['BSA_m2'].notna().sum()}例有BSA值)")
        
        # 验证指数化值范围
        print(f"\n【指数化值验证】")
        print(f"  LVEDVi范围: {left_idx_df['LVEDVi'].min():.1f} - {left_idx_df['LVEDVi'].max():.1f} mL/m²")
        print(f"  正常范围应为: 50-90 mL/m²")
        print(f"  RVEDVi范围: {right_idx_df['RVEDVi'].min():.1f} - {right_idx_df['RVEDVi'].max():.1f} mL/m²")
        print(f"  正常范围应为: 60-100 mL/m²")
    else:
        print(f"  ⚠ BSA数据无效，跳过指数化")

    # QC报告
    with open(os.path.join(out_dir, "quality_control_report_v2.json"), "w", encoding="utf-8") as f:
        json.dump(all_qc_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ quality_control_report_v2.json ({len(all_qc_records)}条记录)")
    
    # 四腔完整性
    qc_counts.to_csv(os.path.join(out_dir, "chamber_completeness_qc.csv"),
                     index=False, encoding="utf-8-sig")
    print(f"  ✓ chamber_completeness_qc.csv ({len(qc_counts)}行)")

    print(f"\n" + "=" * 100)
    print(f"完成！")
    print(f"  左心: {len(left_df)}例")
    print(f"  右心: {len(right_df)}例")
    print(f"  输出位置: {out_dir}")
    print("=" * 100)

if __name__ == "__main__":
    main()
