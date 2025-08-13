# no librosa dependency
import numpy as np

# expose manual processing file functions and constants to process OTG.
import py_files.record_process_audio.manual_audio_process as transform
from py_files.fn_cfg import RUN_TRANSFORM_DIAGNOSTICS

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

		# Tunables (lowered for faster cadence)
		self.max_buffer_sec = 10.0   # keep at most ~10 s of audio
		self.tail_guard_sec = 0.25   # don't finalize a segment within 50 ms of buffer tail
		self.min_analyze_sec = 0.3   # don't analyze until we have at least 300 ms of audio

		# Diagnostics counters (optional)
		self._kept = 0
		self._drop_ber = 0
		self._drop_rms = 0
		self._drop_zcr = 0

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
		y = transform.high_pass_filter(self.buf, self.sr)

		# 1) environment & env-specific params
		env = transform.classify_environment(y, self.sr)

		if env == "noisy":
			LOWER, UPPER = transform.NOISY_BAND
			smooth_win   = transform.RMS_SMOOTH_WIN_NOISY
			gap_tol      = transform.GAP_TOL_FRAMES_NOISY
			rms_pct      = transform.NOISY_RMS_PCTL
			zcr_pct      = transform.NOISY_ZCR_PCTL
			ber_band     = transform.NOISY_BER_BAND
			ber_min      = transform.NOISY_BER_MIN
		else:
			LOWER, UPPER = transform.QUIET_BAND
			smooth_win   = transform.RMS_SMOOTH_WIN_QUIET
			gap_tol      = transform.GAP_TOL_FRAMES_QUIET
			rms_pct      = transform.QUIET_RMS_PCTL
			zcr_pct      = transform.QUIET_ZCR_PCTL
			ber_band     = transform.QUIET_BER_BAND
			ber_min      = transform.QUIET_BER_MIN

		# 2) candidate frames → segments (bounded duration)
		frames, thr_low, thr_high, hop_len = transform.detect_breath_frames(y, self.sr, LOWER, UPPER, smooth_win)
		seg_times = transform.frames_to_segments(frames, self.sr, hop_len, gap_tol, transform.BREATH_MIN_SEC, transform.BREATH_MAX_SEC)

		buf_len_sec = self.buf.size / self.sr
		if RUN_TRANSFORM_DIAGNOSTICS:
			print(f"[SEG] complete={len(seg_times)} buf={buf_len_sec:.2f}s", flush=True)

		if not seg_times:
			return

		# 3) adaptive ZCR/RMS for raw gates
		zcr_cut, rms_cut = transform.adaptive_zcr_rms(y, self.sr, thr_low, thr_high, zcr_pct, rms_pct)

		# 4) finalize only completed segments (not touching the tail_guard)
		tail_guard = self.tail_guard_sec

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
			seg = transform.silence_trim_guard(seg, self.sr, top_db=10)
			if seg.size == 0:
				continue

			ber = transform.band_energy_ratio(seg, self.sr, ber_band[0], ber_band[1])
			if ber < ber_min:
				self._drop_ber += 1
				continue

			ok_rms, _ = transform.noise_gate(seg, threshold=rms_cut)
			if not ok_rms:
				self._drop_rms += 1
				continue

			ok_zcr, _ = transform.zcr_gate(seg, max_zcr=zcr_cut)
			if not ok_zcr:
				self._drop_zcr += 1
				continue

			# Normalize then extract HHT features
			peak = float(np.max(np.abs(seg)))
			if peak > 0:
				seg = seg / peak

			feats = transform.extract_hht_features(
				[seg], self.sr,
				sr_target=transform.HHT_SR_TARGET,
				min_sec=transform.HHT_MIN_SEC,
				max_imf=transform.HHT_IMFS
			)

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
			self._kept += 1
			newest_cut_sample = max(newest_cut_sample, s1)

			if RUN_TRANSFORM_DIAGNOSTICS and self._kept > 0 and (self._kept % 5 == 0):  # print every 5 keeps
				print(f"[KEEP] kept={self._kept} drop: BER={self._drop_ber} RMS={self._drop_rms} ZCR={self._drop_zcr}",
					flush=True)

		# 5) Drop everything up to newest emitted end so CPU stays bounded
		if emitted_any and newest_cut_sample > 0:
			drop = newest_cut_sample
			self.buf = self.buf[drop:]
			self.t0_abs += drop / self.sr
			self.last_emitted_sample = 0
		else:
			# No new emits—remember what we've already handled
			self.last_emitted_sample = newest_cut_sample
