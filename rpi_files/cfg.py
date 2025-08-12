# dependencies
import alsaaudio


"""
AUDIO RECORDING
"""
# ---- Requested capture & processing settings ----
REQ_RATE = 16000          # ask ALSA for 16 kHz if possible (ALSA can resample)
REQ_CH   = 2              # request 2 channels (ReSpeaker 2-mic HAT)
PERIOD   = 256            # ALSA period in *frames* per read (≈16 ms at 16 kHz)

# We'll always process at this rate; channel count follows the input (stereo).
PROC_RATE = 16000         # downstream DSP sample rate (fixed)
# NOTE: We'll set PROC_CH dynamically from the actual input channels.

# 24-bit codecs (e.g., WM8960) typically expose samples as 24-in-32 or full 32-bit.
# Try higher resolutions first, then 16-bit.
FORMAT_CANDIDATES = [
	alsaaudio.PCM_FORMAT_S32_LE,   # common for 24-bit I²S in a 32-bit container
	alsaaudio.PCM_FORMAT_S24_LE,   # explicit 24-in-32 little-endian
	alsaaudio.PCM_FORMAT_S16_LE,   # fallback: 16-bit
]

# Prefer software SRC (plughw/sysdefault) at your requested rate; then raw 48 kHz.
DEVICE_CANDIDATES = [
	('plughw:CARD=seeed2micvoicec,DEV=0', REQ_RATE, REQ_CH),  # ALSA can resample
	('sysdefault:CARD=seeed2micvoicec',   REQ_RATE, REQ_CH),
	('hw:CARD=seeed2micvoicec,DEV=0',     48000,    REQ_CH),  # native 48 kHz fallback
]

# ---- Analysis frame settings ----
FRAME_MS = 25             # analysis window length (e.g., 25 ms for speech)
HOP_MS   = 10             # hop between frames (10 ms → 60% overlap at 25 ms)

# ---- Your pipeline hooks (replace with real ones) ----
# All hooks receive frames as float32 with shape [frame_len, ch] (stereo here).
def clean_block(frame_f32, sr):           return frame_f32      # e.g., HPF, denoise (per-channel)
def extract_features(frame_f32, sr):      return {}             # e.g., MFCC/HHT/etc. from stereo
def process_features(features):           return None           # e.g., classifier
def handle_result(result, t_start_sec):   pass                  # e.g., log/emit result

# ---- Diagnostics (enable/disable) ----
RUN_DIAGNOSTICS = True