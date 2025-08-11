# pip install pyalsaaudio numpy
import alsaaudio, numpy as np, queue, threading, time, signal

# ---- Configure your device/stream ----
DEVICE = 'hw:1,0'   # ReSpeaker card:device (use 'plughw:1,0' if ALSA resampling is OK)
RATE   = 16000      # 16 kHz is typical for speech
CH     = 2          # ReSpeaker 2 mics
PERIOD = 256        # DMA/ALSA period in frames (~16 ms @ 16 kHz)

# ---- Analysis frame settings (optional; comment out if you process raw blocks) ----
FRAME_MS = 25
HOP_MS   = 10
FRAME_LEN = int(RATE * FRAME_MS / 1000)   # e.g., 400 at 16 kHz
HOP_LEN   = int(RATE * HOP_MS / 1000)     # e.g., 160 at 16 kHz

# ---- Bring your own pipeline (placeholders) ----
# You said you already have these; import from your code:
# from my_pipeline import clean_block, extract_features, process_features, handle_result
def clean_block(frame_f32, sr):           # placeholder signature
    return frame_f32
def extract_features(frame_f32, sr):      # placeholder signature
    return {}
def process_features(features):           # placeholder signature
    return None
def handle_result(result, t_start_sec):   # placeholder signature
    pass

# ---- Thread-safe queue between capture and processing ----
blocks = queue.Queue(maxsize=32)
stop = False

def _int16_to_float32(x):
    # Convert interleaved int16 [-32768,32767] to float32 [-1,1]
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def capture_thread():
    """ALSA non-blocking capture; pushes NumPy int16 blocks shaped [nframes, CH]."""
    inp = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE,
                        mode=alsaaudio.PCM_NONBLOCK,
                        device=DEVICE)
    inp.setchannels(CH)
    inp.setrate(RATE)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(PERIOD)

    nap = (PERIOD / RATE) / 4.0
    samples_captured = 0
    t0 = time.monotonic()

    while not stop:
        nframes, data = inp.read()  # (frames, bytes); frames==0 if not ready yet
        if nframes > 0:
            block_i16 = np.frombuffer(data, dtype=np.int16).reshape(-1, CH)
            # Timestamp (approx): start time of this block relative to t0
            t_block_start = (samples_captured / RATE)
            samples_captured += nframes

            try:
                blocks.put_nowait((block_i16, t0 + t_block_start))
            except queue.Full:
                # Drop oldest to stay real-time
                _ = blocks.get_nowait()
                blocks.put_nowait((block_i16, t0 + t_block_start))
        else:
            time.sleep(nap)

class Framer:
    """Yields overlapping analysis frames from arbitrary-sized blocks."""
    def __init__(self, frame_len, hop_len, channels):
        self.frame_len = frame_len
        self.hop_len   = hop_len
        self.ch        = channels
        self.buf = np.empty((0, channels), dtype=np.float32)
        self.buf_t0 = None  # absolute start time (seconds) of buf[0]

    def push(self, block_f32, t_block_start):
        """Append block and yield (frame, t_frame_start) tuples."""
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
            frame = self.buf[:self.frame_len]  # [frame_len, ch]
            t_frame_start = self.buf_t0
            out.append((frame.copy(), t_frame_start))

            # Slide buffer by hop
            self.buf = self.buf[self.hop_len:]
            self.buf_t0 += self.hop_len / RATE

        return out

def processing_thread():
    """Consumes capture blocks, optionally frames them, then calls your functions."""
    framer = Framer(FRAME_LEN, HOP_LEN, CH)
    while not stop:
        try:
            block_i16, t_block_start = blocks.get(timeout=0.5)
        except queue.Empty:
            continue

        # Convert to float32 for DSP
        block_f32 = _int16_to_float32(block_i16)

        # ---- OPTION 1: process per DMA block (comment OPTION 2 if you do this) ----
        # cleaned = clean_block(block_f32, RATE)
        # feats   = extract_features(cleaned, RATE)
        # result  = process_features(feats)
        # handle_result(result, t_block_start)

        # ---- OPTION 2: process in standard analysis frames (25 ms / 10 ms hop) ----
        for frame_f32, t_frame_start in framer.push(block_f32, t_block_start):
            cleaned = clean_block(frame_f32, RATE)
            feats   = extract_features(cleaned, RATE)
            result  = process_features(feats)
            handle_result(result, t_frame_start)

def main():
    global stop
    # Clean Ctrl+C exit
    signal.signal(signal.SIGINT, lambda s, f: setattr(globals()['__builtins__'], 'stop', True))

    t_cap = threading.Thread(target=capture_thread, daemon=True)
    t_proc = threading.Thread(target=processing_thread, daemon=True)
    t_cap.start(); t_proc.start()

    try:
        while not stop:
            time.sleep(0.5)
    finally:
        stop = True
        t_cap.join(timeout=1.0)
        t_proc.join(timeout=1.0)

if __name__ == "__main__":
    main()
