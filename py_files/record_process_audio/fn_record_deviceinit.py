# import constants from cfg file
from py_files.fn_cfg import REQ_CH, REQ_RATE, PERIOD

# dependencies
import alsaaudio

# Recording format, that the function will loop through and try.
# W8960 is 24-bit ADC
FORMAT_CANDIDATES = [
	alsaaudio.PCM_FORMAT_S32_LE,
	alsaaudio.PCM_FORMAT_S24_LE,
	alsaaudio.PCM_FORMAT_S16_LE,
]

# Prefer software SRC (plughw/sysdefault) at your requested rate; then raw 48 kHz.
DEVICE_CANDIDATES = [
	('plughw:CARD=seeed2micvoicec,DEV=0', REQ_RATE, REQ_CH),  # ALSA can resample
	('sysdefault:CARD=seeed2micvoicec',   REQ_RATE, REQ_CH),
	('hw:CARD=seeed2micvoicec,DEV=0',     48000,    REQ_CH),  # native 48 kHz fallback
]

def _fmt_name(fmt):
	return {
		getattr(alsaaudio, 'PCM_FORMAT_S16_LE', None): 'S16_LE',
		getattr(alsaaudio, 'PCM_FORMAT_S24_LE', None): 'S24_LE(24-in-32)',
		getattr(alsaaudio, 'PCM_FORMAT_S32_LE', None): 'S32_LE',
	}.get(fmt, str(fmt))

def open_pcm_with_fallback():
	"""
	Try (device, rate, ch) × formats until one opens.
	Returns (pcm, actual_rate, actual_channels, device_string, fmt).
	Used to init audio device for recording. Returns overwritten global cfg.
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