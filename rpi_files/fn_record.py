# pip install pyalsaaudio numpy
import alsaaudio
import numpy as np, queue, threading, time, signal

# ---- Requested capture & processing settings ----
REQ_RATE = 8000          # request this from ALSA
REQ_CH   = 2             # ReSpeaker 2 mics
PERIOD   = 256           # DMA/ALSA period in frames

# We'll always process at this rate/ch, resampling/downmixing if needed.
PROC_RATE = 8000
PROC_CH   = 1            # mono for downstream DSP

# Prefer software SRC (plughw/sysdefault) at your requested rate, then raw 48k
DEVICE_CANDIDATES = [
    ('plughw:CARD=seeed2micvoicec,DEV=0', REQ_RATE, REQ_CH),
    ('sysdefault:CARD=seeed2micvoicec',   REQ_RATE, REQ_CH),
    ('hw:CARD=seeed2micvoicec,DEV=0',     48000,    REQ_CH),  # fallback: native 48 kHz
]

# ---- Analysis frame settings ----
FRAME_MS = 25
HOP_MS   = 10

# ---- Your pipeline hooks (replace with real ones) ----
def clean_block(frame_f32, sr):           return frame_f32
def extract_features(frame_f32, sr):      return {}
def process_features(features):           return None
def handle_result(result, t_start_sec):   pass

# ---- Shared config/state ----
blocks = queue.Queue(maxsize=32)  # int16 capture blocks
stop_evt = threading.Event()
cfg = {'rate_in': None, 'ch_in': None, 'device': None}

def _int16_to_float32(x):
    # Convert interleaved int16 [-32768,32767] to float32 [-1,1]
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def _downmix_to_mono(f32_2d):
    """[N, ch] -> [N] mono by simple average."""
    if f32_2d.ndim == 2 and f32_2d.shape[1] > 1:
        return f32_2d.mean(axis=1)
    elif f32_2d.ndim == 2 and f32_2d.shape[1] == 1:
        return f32_2d[:, 0]
    elif f32_2d.ndim == 1:
        return f32_2d
    else:
        raise ValueError("Unexpected audio shape")

def _resample_linear_1d(x, in_sr, out_sr):
    """Cheap linear resampler good enough for 48k->8k telemetry."""
    if in_sr == out_sr or len(x) == 0:
        return x
    n_out = int(round(len(x) * (out_sr / float(in_sr))))
    if n_out <= 1:
        return np.zeros((0,), dtype=x.dtype)
    t_in  = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out,   endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)

def open_pcm_with_fallback():
    """Try several device/rate combos; return (pcm, rate, ch, dev)."""
    last_err = None
    for dev, rate, ch in DEVICE_CANDIDATES:
        try:
            pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                mode=alsaaudio.PCM_NONBLOCK,
                device=dev,
                channels=ch,
                rate=rate,
                format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=PERIOD,
            )
            print(f"[ALSA] Using {dev} rate={rate} ch={ch}")
            return pcm, rate, ch, dev
        except alsaaudio.ALSAAudioError as e:
            print(f"[ALSA] Try {dev} rate={rate} ch={ch} -> {e}")
            last_err = e
    raise RuntimeError(f"No ALSA config worked: {last_err}")

class Framer:
    """Yields overlapping analysis frames from arbitrary-sized blocks."""
    def __init__(self, frame_len, hop_len, channels, sr):
        self.frame_len = frame_len
        self.hop_len   = hop_len
        self.ch        = channels
        self.sr        = sr
        self.buf = np.empty((0, channels), dtype=np.float32)
        self.buf_t0 = None  # absolute start time (seconds) of buf[0]

    def push(self, block_f32, t_block_start):
        out = []
        if self.buf.size == 0:
            self.buf_t0 = t_block_start

        # Append
        if self.buf.shape[0] == 0:
            self.buf = block_f32
        else:
            self.buf = np.vstack((self.buf, block_f32))

        # Emit frames while we have enough samples
        while self.buf.shape[0] >= self.frame_len:
            frame = self.buf[:self.frame_len]              # [frame_len, ch]
            t_frame_start = self.buf_t0
            out.append((frame.copy(), t_frame_start))
            # Slide buffer by hop
            self.buf = self.buf[self.hop_len:]
            self.buf_t0 += self.hop_len / self.sr
        return out

def capture_thread():
    """ALSA non-blocking capture; pushes NumPy int16 blocks shaped [nframes, ch]."""
    pcm, rate_in, ch_in, dev = open_pcm_with_fallback()
    cfg['rate_in'], cfg['ch_in'], cfg['device'] = rate_in, ch_in, dev

    nap = (PERIOD / rate_in) / 4.0
    samples_captured = 0
    t0 = time.monotonic()

    while not stop_evt.is_set():
        try:
            nframes, data = pcm.read()  # (frames, bytes); frames==0 if not ready yet
        except alsaaudio.ALSAAudioError:
            time.sleep(nap)
            continue

        if nframes > 0 and data:
            block_i16 = np.frombuffer(data, dtype=np.int16)
            # Defensive reshape (drop trailing partial samples if any)
            n_samp = (block_i16.size // ch_in) * ch_in
            if n_samp != block_i16.size:
                block_i16 = block_i16[:n_samp]
            if n_samp == 0:
                continue
            block_i16 = block_i16.reshape(-1, ch_in)

            # Timestamp: start time of this block relative to t0
            t_block_start = (samples_captured / rate_in)
            # Advance by the frames we actually kept
            samples_captured += block_i16.shape[0]

            try:
                blocks.put_nowait((block_i16, t0 + t_block_start))
            except queue.Full:
                # Drop oldest block to make room (bounded latency)
                try:
                    _ = blocks.get_nowait()
                except queue.Empty:
                    pass
                blocks.put_nowait((block_i16, t0 + t_block_start))
        else:
            time.sleep(nap)

def processing_thread():
    """Consumes capture blocks, resamples/downmixes, frames, then calls your functions."""
    # Wait until capture thread sets actual cfg
    while not stop_evt.is_set() and (cfg['rate_in'] is None or cfg['ch_in'] is None):
        time.sleep(0.01)
    if stop_evt.is_set():
        return

    rate_in = cfg['rate_in']
    # We will frame at PROC_RATE / PROC_CH regardless of input
    frame_len = int(PROC_RATE * FRAME_MS / 1000)
    hop_len   = int(PROC_RATE * HOP_MS   / 1000)
    framer = Framer(frame_len, hop_len, PROC_CH, PROC_RATE)

    while not stop_evt.is_set():
        try:
            block_i16, t_block_start = blocks.get(timeout=0.5)
        except queue.Empty:
            continue

        # Convert to float32 for DSP, shape [N, ch_in]
        block_f32 = _int16_to_float32(block_i16)

        # Downmix to mono -> [N]
        mono = _downmix_to_mono(block_f32)

        # Resample if needed to PROC_RATE -> [N_proc]
        mono_proc = _resample_linear_1d(mono, rate_in, PROC_RATE)

        # Ensure shape [N_proc, 1] for the framer
        if mono_proc.size == 0:
            continue
        mono_proc_2d = mono_proc.reshape(-1, 1)

        # Frame and process
        for frame_f32, t_frame_start in framer.push(mono_proc_2d, t_block_start):
            cleaned = clean_block(frame_f32, PROC_RATE)
            feats   = extract_features(cleaned, PROC_RATE)
            result  = process_features(feats)
            handle_result(result, t_frame_start)

def main():
    def on_sigint(sig, frm):
        stop_evt.set()
    try:
        signal.signal(signal.SIGINT, on_sigint)
        signal.signal(signal.SIGTERM, on_sigint)
    except Exception:
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

if __name__ == "__main__":
    main()
