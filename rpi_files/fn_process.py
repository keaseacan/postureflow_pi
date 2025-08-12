# dependencies
import os
import shutil
import numpy as np
import librosa
import scipy.signal as signal
import pandas as pd
import soundfile as sf
from PyEMD import EMD
from scipy.signal import hilbert, butter, sosfilt

# =========================
# High-level toggles
# =========================
CLEAR_OLD_SEGMENTS = False     # wipe per-file output folder before saving new segments
SAVE_SEGMENTS      = False     # save each kept breath as a .wav
OUTPUT_XLSX        = "breathing_features4.xlsx" # output features to this excel file
INPUT_DIR          = "input_audio4"        # your input folder
SEG_ROOT           = "breath_segments"    # where per-file segment folders go

# =========================
# Core settings
# =========================
HPF_CUTOFF_HZ = 70            # global high-pass to remove rumble; DO NOT low-pass before HHT

# Frameing for detection
FRAME_MS = 20
HOP_MS   = 10

# Breath duration guardrails (seconds)
BREATH_MIN_SEC = 0.35
BREATH_MAX_SEC = 1.50

# Smoothed RMS for candidate detection
RMS_SMOOTH_WIN_QUIET = 9
RMS_SMOOTH_WIN_NOISY = 11

# Gap tolerance (frames) when merging candidate runs
GAP_TOL_FRAMES_QUIET = 3
GAP_TOL_FRAMES_NOISY = 7

# Fixed RMS band percentiles (for candidate frames), per environment
QUIET_BAND = (10, 35)   # more permissive
NOISY_BAND = (8, 30)    # tighter band in noise

# Adaptive per-file gates (computed from in-band frames)
# Higher ZCR percentile = more tolerant in noise; lower RMS percentile = keeps quiet breaths
QUIET_RMS_PCTL = 20
NOISY_RMS_PCTL = 25
QUIET_ZCR_PCTL = 90
NOISY_ZCR_PCTL = 90
ZCR_MAX_CAP    = 0.50
RMS_MIN_FLOOR  = 1e-5

# ---- BER gate (Spectral band energy ratio) ----
# Choose band per environment and a minimum BER threshold to pass.
# Start: quiet 60–1500 Hz, noisy 60–1200 Hz
QUIET_BER_BAND = (60, 1500)
NOISY_BER_BAND = (60, 1200)
QUIET_BER_MIN  = 0.30   # try 0.25–0.30 in quiet
NOISY_BER_MIN  = 0.60   # try 0.35–0.45 in noisy
# ==============================================

# HHT / EMD settings
HHT_SR_TARGET = 16000        # resample segments for stable EMD
HHT_MIN_SEC   = 0.25         # pad segments shorter than this for EMD
HHT_IMFS      = 9            # always create 9 features (pad with 0 if fewer returned)

# -------------------- Basics --------------------
def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr

def high_pass_filter(y, sr, cutoff=HPF_CUTOFF_HZ):
    sos = signal.butter(10, cutoff, 'hp', fs=sr, output='sos')
    return signal.sosfilt(sos, y)

# ------------- Environment heuristic -------------
def classify_environment(y, sr):
    """Heuristic: use RMS/ZCR/flatness medians to flag noisy vs quiet."""
    frame_len = int(FRAME_MS * sr / 1000)
    hop_len   = int(HOP_MS   * sr / 1000)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    if len(rms) == 0:
        return "quiet"
    rms10 = np.percentile(rms, 10)
    noisy_like = (np.median(rms) > 1.6 * rms10) or (np.median(zcr) > 0.12)
    return "noisy" if noisy_like else "quiet"

# ------- Candidate detection w/ smoothed RMS ------
def detect_breath_frames(y, sr, lower_pctl, upper_pctl, smooth_win):
    frame_len = int(FRAME_MS * sr / 1000)
    hop_len   = int(HOP_MS   * sr / 1000)
    energy = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    if len(energy) == 0:
        return np.array([]), 0.0, 0.0, hop_len

    energy_s = np.convolve(energy, np.ones(smooth_win)/smooth_win, mode="same") if len(energy) >= smooth_win else energy
    thr_low  = np.percentile(energy_s, lower_pctl)
    thr_high = np.percentile(energy_s, upper_pctl)
    cand = (energy_s > thr_low) & (energy_s < thr_high)
    breath_frames = np.where(cand)[0]
    kept_pct = (len(breath_frames) / len(energy_s)) * 100
    print(f"[Detect] Band {lower_pctl}-{upper_pctl}th, smooth={smooth_win} → {len(breath_frames)} frames ({kept_pct:.1f}%)")
    return breath_frames, thr_low, thr_high, hop_len

# --------------- Build segments -------------------
def frames_to_segments(breath_frames, sr, hop_len, gap_tol_frames, min_sec, max_sec):
    if breath_frames.size == 0:
        return []
    segs = []
    s = e = breath_frames[0]
    for idx in breath_frames[1:]:
        if idx <= e + 1 + gap_tol_frames:
            e = idx
        else:
            segs.append((s, e)); s = e = idx
    segs.append((s, e))

    hop_sec   = librosa.frames_to_time(1, sr=sr, hop_length=hop_len)
    min_frames = max(1, int(np.round(min_sec / hop_sec)))
    max_frames = max(1, int(np.round(max_sec / hop_sec)))

    seg_times = []
    for s, e in segs:
        length = e - s + 1
        if length < min_frames or length > max_frames:
            continue
        start_t = librosa.frames_to_time(s, sr=sr, hop_length=hop_len)
        end_t   = librosa.frames_to_time(e, sr=sr, hop_length=hop_len)
        seg_times.append((start_t, end_t))
    return seg_times

# --------------- Spectral helpers -----------------
def band_energy_ratio(y, sr, f_lo, f_hi):
    e_total = np.sum(y**2) + 1e-12
    sos = butter(4, [f_lo, f_hi], btype='band', fs=sr, output='sos')
    yb = sosfilt(sos, y)
    e_band = np.sum(yb**2)
    return float(e_band / e_total)

# --------------- RAW gates (per segment) ----------
def silence_trim_guard(segment, top_db=10):
    trimmed, _ = librosa.effects.trim(segment, top_db=top_db)
    return segment if len(trimmed) < 0.2 * len(segment) else trimmed

def noise_gate(segment, threshold):
    rms = np.sqrt(np.mean(segment**2))
    return (rms >= threshold), rms

def zcr_gate(segment, max_zcr):
    z = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
    return (z <= max_zcr), z

# ---- Adaptive ZCR/RMS from in-band frames (percentiles) ----
def adaptive_zcr_rms(y, sr, thr_low, thr_high, zcr_pct, rms_pct):
    frame_len = int(FRAME_MS * sr / 1000)
    hop_len   = int(HOP_MS   * sr / 1000)
    energy = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr    = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    if len(energy) == 0:
        return 0.35, 0.0002
    energy_s = np.convolve(energy, np.ones(9)/9, mode="same") if len(energy) >= 9 else energy
    in_band  = (energy_s > thr_low) & (energy_s < thr_high)

    z_in = zcr[in_band]
    e_in = energy[in_band]

    if z_in.size:
        z_cut = float(np.percentile(z_in, zcr_pct))
        z_cut = min(z_cut, ZCR_MAX_CAP)
    else:
        z_cut = 0.35

    if e_in.size:
        r_cut = float(np.percentile(e_in, rms_pct))
        r_cut = max(r_cut, RMS_MIN_FLOOR)
    else:
        r_cut = 0.0002

    print(f"[Adaptive] ZCR_cut={z_cut:.3f} (p{zcr_pct}), RMS_cut={r_cut:.6f} (p{rms_pct})")
    return z_cut, r_cut

# ---------------- HHT features --------------------
def extract_hht_features(segments, sr, sr_target=HHT_SR_TARGET, min_sec=HHT_MIN_SEC, max_imf=HHT_IMFS):
    emd = EMD()
    out = []
    min_len = int(sr_target * min_sec)

    for seg in segments:
        y = librosa.resample(seg.astype(np.float64), orig_sr=sr, target_sr=sr_target, res_type="kaiser_best")
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)), mode='constant')
        try:
            imfs = emd.emd(y, max_imf=max_imf)
        except Exception:
            y_sm = np.convolve(y, np.ones(5)/5, mode="same") if len(y) > 5 else y
            imfs = emd.emd(y_sm, max_imf=max_imf)

        feats = []
        K = imfs.shape[0] if imfs.ndim == 2 else 0
        for k in range(max_imf):
            if k < K:
                analytic = hilbert(imfs[k, :])
                feats.append(float(np.mean(np.abs(analytic))))
            else:
                feats.append(0.0)
        out.append(feats)

    return np.array(out)

# ---------------- Main per-file -------------------
def process_file(path):
    y, sr = load_audio(path)
    y = high_pass_filter(y, sr)

    # 1) environment profile
    env = classify_environment(y, sr)
    if env == "noisy":
        LOWER, UPPER = NOISY_BAND
        smooth_win   = RMS_SMOOTH_WIN_NOISY
        gap_tol      = GAP_TOL_FRAMES_NOISY
        rms_pct      = NOISY_RMS_PCTL
        zcr_pct      = NOISY_ZCR_PCTL
        ber_band     = NOISY_BER_BAND
        ber_min      = NOISY_BER_MIN
    else:
        LOWER, UPPER = QUIET_BAND
        smooth_win   = RMS_SMOOTH_WIN_QUIET
        gap_tol      = GAP_TOL_FRAMES_QUIET
        rms_pct      = QUIET_RMS_PCTL
        zcr_pct      = QUIET_ZCR_PCTL
        ber_band     = QUIET_BER_BAND
        ber_min      = QUIET_BER_MIN

    print(f"\n▶ Env='{env}' → band={LOWER}-{UPPER}th, smooth={smooth_win}, gap_tol={gap_tol}, "
          f"RMS%={rms_pct}, ZCR%={zcr_pct}, BER≥{ber_min} ({ber_band[0]}–{ber_band[1]} Hz)")

    # 2) candidate frames from smoothed RMS
    frames, thr_low, thr_high, hop_len = detect_breath_frames(y, sr, LOWER, UPPER, smooth_win)

    # 3) contiguous segments w/ duration constraints
    seg_times = frames_to_segments(frames, sr, hop_len, gap_tol, BREATH_MIN_SEC, BREATH_MAX_SEC)
    print(f"📦 {len(seg_times)} candidate segments in {os.path.basename(path)}")

    # 4) adaptive ZCR/RMS cutoffs from in-band frames
    zcr_cut, rms_cut = adaptive_zcr_rms(y, sr, thr_low, thr_high, zcr_pct, rms_pct)

    # 5) extract / gate / save
    seg_wavs = []
    durations_ms = []
    save_dir = os.path.join(SEG_ROOT, os.path.splitext(os.path.basename(path))[0])
    if CLEAR_OLD_SEGMENTS and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    if SAVE_SEGMENTS:
        os.makedirs(save_dir, exist_ok=True)

    kept_idx = 0
    removed_ber = removed_rms = removed_zcr = 0

    for i, (t0, t1) in enumerate(seg_times):
        seg = y[int(t0 * sr): int(t1 * sr)]
        if seg.size == 0:
            continue

        # trim edges (guardrail reverts over-trim)
        seg = silence_trim_guard(seg, top_db=10)
        if seg.size == 0:
            continue

        # --- BER gate (ENV-SPECIFIC band) ---
        ber = band_energy_ratio(seg, sr, ber_band[0], ber_band[1])
        if ber < ber_min:
            removed_ber += 1
            continue

        # --- RAW RMS gate ---
        ok_rms, rms = noise_gate(seg, threshold=rms_cut)
        if not ok_rms:
            removed_rms += 1
            continue

        # --- RAW ZCR gate ---
        ok_zcr, z = zcr_gate(seg, max_zcr=zcr_cut)
        if not ok_zcr:
            removed_zcr += 1
            continue

        # normalize and keep
        peak = np.max(np.abs(seg))
        if peak > 0:
            seg = seg / peak

        kept_idx += 1
        seg_wavs.append(seg)
        durations_ms.append((len(seg) / sr) * 1000.0)
        if SAVE_SEGMENTS:
            outp = os.path.join(save_dir, f"breath_{kept_idx}.wav")
            sf.write(outp, seg, sr)

    print(f"[Counts] BER={removed_ber} | RMS={removed_rms} | ZCR={removed_zcr} | kept={kept_idx}")

    # 6) HHT features
    feats = extract_hht_features(seg_wavs, sr, sr_target=HHT_SR_TARGET,
                                 min_sec=HHT_MIN_SEC, max_imf=HHT_IMFS)
    return feats, durations_ms, env

# ---------------- Batch runner --------------------
def main():
    rows = []
    for fn in os.listdir(INPUT_DIR):
        if not fn.lower().endswith(".wav"):
            continue
        fp = os.path.join(INPUT_DIR, fn)
        print(f"\n🎧 Processing {fn} ...")
        try:
            feats, durs, env = process_file(fp)
            for i, fv in enumerate(feats):
                row = {
                    "File": fn,
                    "EnvProfile": env,
                    "Breath_Index": i + 1,
                    "Duration_ms": durs[i] if i < len(durs) else None,
                    **{f"IMF_{k+1}": float(fv[k]) for k in range(HHT_IMFS)}
                }
                rows.append(row)
        except PermissionError as e:
            print(f"⚠️ Permission error (is {OUTPUT_XLSX} open?): {e}")
        except Exception as e:
            print(f"❌ Error processing {fn}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        try:
            df.to_excel(OUTPUT_XLSX, index=False)
            print(f"\n📄 Features saved to {OUTPUT_XLSX}")
        except PermissionError as e:
            print(f"⚠️ Could not write {OUTPUT_XLSX} (close it if open): {e}")
    else:
        print("ℹ️ No rows to save.")

if __name__ == "__main__":
    main()
