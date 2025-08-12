# rt_breath_hooks.py
import numpy as np
import fn_process_haziq as P  # reuse your functions and constants

class RealTimeBreathDetector:
    """
    Incremental breath segmenter using your fn_process logic.
    Feed frames via .push(frame_f32[N, ch], t_abs_start_seconds).
    Calls on_segment(result_dict) whenever a complete, gated segment is found.
    """
    def __init__(self, sr_proc: int, on_segment):
        self.sr = sr_proc
        self.on_segment = on_segment

        # Rolling mono buffer (for detection/gating/HHT). Absolute time for buf[0] in seconds.
        self.buf = np.empty((0,), dtype=np.float32)
        self.t0_abs = None

        # Bookkeeping to avoid duplicate emits
        self.last_emitted_sample = 0  # sample idx within current buf of last emitted end

        # Tunables
        self.max_buffer_sec = 20.0    # keep at most ~20 s of audio
        self.tail_guard_sec = 0.15    # don’t finalize a segment that ends within 150 ms of buffer tail
        self.min_analyze_sec = 1.0    # don’t analyze until we have at least this much audio

    @staticmethod
    def _downmix_mono(x):
        return x.mean(axis=1).astype(np.float32, copy=False) if x.ndim == 2 else x.astype(np.float32, copy=False)

    def push(self, frame_f32: np.ndarray, t_frame_abs: float):
        """Append a new frame and try to detect/emit completed segments."""
        mono = self._downmix_mono(frame_f32)

        if self.buf.size == 0:
            self.t0_abs = t_frame_abs

        # Append to rolling buffer
        self.buf = np.concatenate([self.buf, mono])

        # Trim to max_buffer_sec while keeping time/bookkeeping consistent
        max_samples = int(self.max_buffer_sec * self.sr)
        if self.buf.size > max_samples:
            drop = self.buf.size - max_samples
            self.buf = self.buf[drop:]
            self.t0_abs += drop / self.sr
            self.last_emitted_sample = max(0, self.last_emitted_sample - drop)

        # Only analyze when we have enough audio
        if self.buf.size < int(self.min_analyze_sec * self.sr):
            return

        self._analyze_and_emit()

    def _analyze_and_emit(self):
        y = P.high_pass_filter(self.buf, self.sr)

        # 1) environment & env-specific params
        env = P.classify_environment(y, self.sr)

        if env == "noisy":
            LOWER, UPPER = P.NOISY_BAND
            smooth_win   = P.RMS_SMOOTH_WIN_NOISY
            gap_tol      = P.GAP_TOL_FRAMES_NOISY
            rms_pct      = P.NOISY_RMS_PCTL
            zcr_pct      = P.NOISY_ZCR_PCTL
            ber_band     = P.NOISY_BER_BAND
            ber_min      = P.NOISY_BER_MIN
        else:
            LOWER, UPPER = P.QUIET_BAND
            smooth_win   = P.RMS_SMOOTH_WIN_QUIET
            gap_tol      = P.GAP_TOL_FRAMES_QUIET
            rms_pct      = P.QUIET_RMS_PCTL
            zcr_pct      = P.QUIET_ZCR_PCTL
            ber_band     = P.QUIET_BER_BAND
            ber_min      = P.QUIET_BER_MIN

        # 2) candidate frames → segments (bounded duration)
        frames, thr_low, thr_high, hop_len = P.detect_breath_frames(y, self.sr, LOWER, UPPER, smooth_win)
        seg_times = P.frames_to_segments(frames, self.sr, hop_len, gap_tol, P.BREATH_MIN_SEC, P.BREATH_MAX_SEC)

        if not seg_times:
            return

        # 3) adaptive ZCR/RMS for raw gates
        zcr_cut, rms_cut = P.adaptive_zcr_rms(y, self.sr, thr_low, thr_high, zcr_pct, rms_pct)

        # 4) finalize only completed segments (not touching the tail_guard)
        tail_guard = self.tail_guard_sec
        buf_len_sec = self.buf.size / self.sr

        emitted_any = False
        newest_cut_sample = self.last_emitted_sample

        for (t0, t1) in seg_times:
            if t1 > (buf_len_sec - tail_guard):
                # likely still ongoing; wait for more audio
                continue

            s0 = int(t0 * self.sr)
            s1 = int(t1 * self.sr)
            if s1 <= self.last_emitted_sample:
                continue  # already emitted earlier

            seg = y[s0:s1]
            if seg.size == 0:
                continue

            # Reuse your per-segment gates
            seg = P.silence_trim_guard(seg, self.sr, top_db=10)
            if seg.size == 0:
                continue

            ber = P.band_energy_ratio(seg, self.sr, ber_band[0], ber_band[1])
            if ber < ber_min:
                continue

            ok_rms, _ = P.noise_gate(seg, threshold=rms_cut)
            if not ok_rms:
                continue

            ok_zcr, _ = P.zcr_gate(seg, max_zcr=zcr_cut)
            if not ok_zcr:
                continue

            # Normalize then extract HHT features
            peak = float(np.max(np.abs(seg)))
            if peak > 0:
                seg = seg / peak

            feats = P.extract_hht_features([seg], self.sr,
                                           sr_target=P.HHT_SR_TARGET,
                                           min_sec=P.HHT_MIN_SEC,
                                           max_imf=P.HHT_IMFS)
            # Compose result
            result = {
                "EnvProfile": env,
                "Duration_ms": (len(seg) / self.sr) * 1000.0,
                "IMF": feats[0].tolist(),
                "t_abs_start": self.t0_abs + t0,
            }
            # Emit
            self.on_segment(result)
            emitted_any = True
            newest_cut_sample = max(newest_cut_sample, s1)

        # 5) Drop everything up to newest emitted end so CPU stays bounded
        if emitted_any and newest_cut_sample > 0:
            drop = newest_cut_sample
            self.buf = self.buf[drop:]
            self.t0_abs += drop / self.sr
            self.last_emitted_sample = 0
        else:
            # No new emits—remember what we've already handled
            self.last_emitted_sample = newest_cut_sample


# --- Thin hook wrappers you can import in your recorder if you prefer that pattern ---
# (Optional: If you’d rather call detector.push() directly in your loop, you can ignore these.)
_DETECTOR = None

def init_realtime_detector(sr_proc: int, on_segment):
    global _DETECTOR
    _DETECTOR = RealTimeBreathDetector(sr_proc, on_segment)

def clean_block(frame_f32, sr):     # runs before feature extraction
    return frame_f32

def extract_features(frame_f32, sr):
    # We piggyback to get the frame into the detector; return a dummy.
    if _DETECTOR is None:
        return None
    # Timestamp is handled by recorder; this function only receives the frame.
    # We'll do the actual push in process_features where we have the timestamp (passed through the recorder).
    return frame_f32

# We’ll pass the frame + timestamp via the recorder’s call site.
def process_features(payload):
    # no-op; the recorder will call detector.push(frame, t_frame_start) directly
    return None

def handle_result(result, t_start_sec):
    # The detector will call on_segment(result) itself. The per-frame call can be ignored.
    pass
