# other py file dependencies
from fn_record_helpers import _resample_linear, _bytes_to_float32
from fn_record_debug import _fmt_name, Diagnostics
from fn_record_buffer import Framer
from fn_process_breath import RealTimeBreathDetector
from cfg import REQ_CH, REQ_RATE, PERIOD
from cfg import PROC_RATE
from cfg import FORMAT_CANDIDATES, DEVICE_CANDIDATES
from cfg import FRAME_MS, HOP_MS
from cfg import RUN_DIAGNOSTICS

# pip install pyalsaaudio numpy
import alsaaudio
import numpy as np, queue, threading, time, signal

# -----------------------------------------------------------------------------
# Real-time audio pipeline overview (STEREO preserved)
#   ALSA capture (non-blocking, native bit depth: 16 / 24-in-32 / 32) → bounded Queue
#   → processing thread:
#     bytes→float32 (preserve ADC resolution) → (NO downmix; keep [L,R]) →
#     resample both channels to PROC_RATE → frame into 25 ms windows with 10 ms hop →
#     call your hooks with shape [frame_len, 2].
#   If processing is slower than capture, oldest blocks are dropped to keep
#   latency bounded (no unbounded buffering).
# -----------------------------------------------------------------------------

# ---- Your pipeline hooks (replace with real ones) ----
# All hooks receive frames as float32 with shape [frame_len, ch] (stereo here).
def clean_block(frame_f32, sr):           return frame_f32      # e.g., HPF, denoise (per-channel)
def extract_features(frame_f32, sr):      return {}             # e.g., MFCC/HHT/etc. from stereo
def process_features(features):           return None           # e.g., classifier
def handle_result(result, t_start_sec):   pass                  # e.g., log/emit result

# ---- Shared config/state ----
# Queue carries float32 blocks (resolution-agnostic) + absolute (monotonic-based) timestamp.
blocks = queue.Queue(maxsize=32)  # bounded to cap latency under backpressure
stop_evt = threading.Event()

# rate_in = sampling rate, based on REQ_RATE if not what the device allows from FORMAT_CANDIDATES
# ch_in = recording channels
# device = what is chosen from DEVICE_CANDIDATES
# fmt = the chosen format from FORMAT_CANDIDATES
cfg = {'rate_in': None, 'ch_in': None, 'device': None, 'fmt': None}  # filled after device opens

def main():
	"""
	Install Ctrl+C handlers, start capture & processing threads as daemons,
	and keep the main thread alive until stop_evt is set.
	"""
	def on_sigint(sig, frm):
		stop_evt.set()
	try:
		signal.signal(signal.SIGINT, on_sigint)
		signal.signal(signal.SIGTERM, on_sigint)
		if RUN_DIAGNOSTICS: print("signal work")
	except Exception:
		if RUN_DIAGNOSTICS: print("signal not work")
		pass

	t_cap = threading.Thread(target=capture_thread, daemon=True)
	t_proc = threading.Thread(target=processing_thread, daemon=True)
	t_cap.start(); t_proc.start()

	try:
		while not stop_evt.is_set():
			time.sleep(0.5)
	finally:
		stop_evt.set()
		t_cap.join(timeout=1.0)
		t_proc.join(timeout=1.0)
		# NOTE: You can also close ALSA PCM explicitly in capture_thread on exit.

# used to init audio device
# returned pcm overwrites global cfg
def open_pcm_with_fallback():
	"""
	Try (device, rate, ch) × formats until one opens.
	Returns (pcm, actual_rate, actual_channels, device_string, fmt).
	"""
	last_err = None
	for dev, rate, ch in DEVICE_CANDIDATES:
		for fmt in FORMAT_CANDIDATES:
			try:
				pcm = alsaaudio.PCM(
					type=alsaaudio.PCM_CAPTURE,
					mode=alsaaudio.PCM_NONBLOCK,
					device=dev,
					channels=ch,
					rate=rate,
					format=fmt,
					periodsize=PERIOD,
				)
				print(f"[ALSA] Using {dev} rate={rate} ch={ch} fmt={_fmt_name(fmt)}")
				return pcm, rate, ch, dev, fmt
			except alsaaudio.ALSAAudioError as e:
				last_err = e  # keep last error for context
	raise RuntimeError(f"No ALSA config worked: {last_err}")

def capture_thread():
	"""
	Opens ALSA PCM and continuously reads non-blocking chunks.
	Each successful read:
	raw bytes → float32 [nframes, ch] at native resolution (16/24/32-bit)
	compute the block start time relative to stream start
	push (block_f32, absolute_time) into the queue
	If the queue is full, drop the oldest to bound latency.
	"""
	pcm, rate_in, ch_in, dev, fmt = open_pcm_with_fallback()
	cfg['rate_in'], cfg['ch_in'], cfg['device'], cfg['fmt'] = rate_in, ch_in, dev, fmt

	nap = (PERIOD / rate_in) / 4.0  # poll several times per period
	samples_captured = 0            # count of accepted frames, for timestamps
	t0 = time.monotonic()           # monotonic reference at stream start

	while not stop_evt.is_set():
		try:
			nframes, data = pcm.read()  # non-blocking; nframes==0 → no data yet
		except alsaaudio.ALSAAudioError:
			time.sleep(nap); continue

		if nframes > 0 and data:
			block_f32 = _bytes_to_float32(data, fmt, ch_in)
			if block_f32.size == 0:
				continue

			t_block_start = (samples_captured / rate_in)  # seconds since t0
			samples_captured += block_f32.shape[0]        # advance by kept frames

			try:
				blocks.put_nowait((block_f32, t0 + t_block_start))
			except queue.Full:
				# Drop oldest to cap latency, then push
				try:
					_ = blocks.get_nowait()
				except queue.Empty: 
					pass
				blocks.put_nowait((block_f32, t0 + t_block_start))
		else:
			time.sleep(nap)
	"""
	def processing_thread():
	Consumes capture blocks from the queue (float32, stereo), resamples both channels
	to PROC_RATE, then uses the Framer to emit 25 ms / 10 ms frames.
	For each frame (shape [frame_len, 2]), call the user hooks in order.
	# Wait for capture config
	while not stop_evt.is_set() and (cfg['rate_in'] is None or cfg['ch_in'] is None):
		time.sleep(0.01)
	if stop_evt.is_set():
		return

	rate_in = cfg['rate_in']
	ch_proc = cfg['ch_in']  # keep the same channel count as the input (stereo)

	# Frame sizes in samples at processing rate
	frame_len = int(PROC_RATE * FRAME_MS / 1000)  # 25 ms @ 16 kHz → 400
	hop_len   = int(PROC_RATE * HOP_MS   / 1000)  # 10 ms @ 16 kHz → 160
	framer = Framer(frame_len, hop_len, ch_proc, PROC_RATE)

	# Diagnostics setup
	diag = Diagnostics(PROC_RATE, ch_proc, hop_len) if RUN_DIAGNOSTICS else None
	if RUN_DIAGNOSTICS:
		diag.on_open(cfg)

	while not stop_evt.is_set():
		try:
			block_f32, t_block_start = blocks.get(timeout=0.5)  # [N, ch_proc], abs time
		except queue.Empty:
			continue

		# Resample both channels to fixed processing rate → [N_proc, ch_proc]
		block_proc = _resample_linear(block_f32, rate_in, PROC_RATE)
		if block_proc.size == 0:
			continue

		# Frame & process
		for frame_f32, t_frame_start in framer.push(block_proc, t_block_start):
			if RUN_DIAGNOSTICS:
				diag.check_frame(frame_f32, t_frame_start)

			cleaned = clean_block(frame_f32, PROC_RATE)           # e.g., HPF per channel
			feats   = extract_features(cleaned, PROC_RATE)        # stereo-aware features
			result  = process_features(feats)                     # your model
			handle_result(result, t_frame_start)                  # log/emit
"""

	def processing_thread():
		# Wait for capture config
		while not stop_evt.is_set() and (cfg['rate_in'] is None or cfg['ch_in'] is None):
				time.sleep(0.01)
		if stop_evt.is_set():
				return

		rate_in = cfg['rate_in']
		ch_proc = cfg['ch_in']  # keep stereo for capture; we downmix in detector

		frame_len = int(PROC_RATE * FRAME_MS / 1000)
		hop_len   = int(PROC_RATE * HOP_MS   / 1000)
		framer = Framer(frame_len, hop_len, ch_proc, PROC_RATE)

		diag = Diagnostics(PROC_RATE, ch_proc, hop_len) if RUN_DIAGNOSTICS else None
		if RUN_DIAGNOSTICS:
				diag.on_open(cfg)

		# Instantiate the detector and tell it how to report results
		def on_segment(res):
			# res = {"EnvProfile", "Duration_ms", "IMF":[...], "t_abs_start": seconds}
			# Replace this with your logger / model / BLE emit, etc.
			print(f"[BREATH] t={res['t_abs_start']:.3f}s, dur={res['Duration_ms']:.1f} ms, env={res['EnvProfile']}, IMF1={res['IMF'][0]:.4f}")

			detector = RealTimeBreathDetector(PROC_RATE, on_segment=on_segment)
			while not stop_evt.is_set():
				try:
					block_f32, t_block_start = blocks.get(timeout=0.5)
				except queue.Empty:
					continue

				block_proc = _resample_linear(block_f32, rate_in, PROC_RATE)
				if block_proc.size == 0:
					continue

				for frame_f32, t_frame_start in framer.push(block_proc, t_block_start):
					if RUN_DIAGNOSTICS:
						diag.check_frame(frame_f32, t_frame_start)

					# Feed the detector; no need to go through the placeholder hook chain
					detector.push(frame_f32, t_frame_start)

if __name__ == "__main__":
	main()