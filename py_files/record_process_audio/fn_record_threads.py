# functions
from py_files.record_process_audio.fn_record_helpers import cfg, _bytes_to_float32, _resample_poly
from py_files.record_process_audio.fn_record_deviceinit import open_pcm_with_fallback
from py_files.record_process_audio.fn_record_buffer import stop_evt, blocks, Framer
from py_files.record_process_audio.fn_record_debug import Diagnostics
from py_files.record_process_audio.fn_process_breath import RealTimeBreathDetector

# constants from cfg file
from py_files.fn_cfg import PERIOD, PROC_RATE, FRAME_MS, HOP_MS
from py_files.fn_cfg import RUN_RECORD_DIAGNOSTICS, RUN_TRANSFORM_DIAGNOSTICS

# dependencies
import alsaaudio
import queue
import time

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
                try:
                    blocks.put_nowait((block_f32, t0 + t_block_start))
                except queue.Full:
                    pass
        else:
            time.sleep(nap)

def processing_thread(emit_queue: queue.Queue | None = None):
    # Wait for capture config from capture_thread
    while not stop_evt.is_set() and (cfg['rate_in'] is None or cfg['ch_in'] is None):
        time.sleep(0.01)
    if stop_evt.is_set():
        return

    rate_in = cfg['rate_in']
    ch_proc = cfg['ch_in']  # keep stereo for capture; downmix happens in detector if needed

    # Frame sizes in samples at processing rate
    frame_len = int(PROC_RATE * FRAME_MS / 1000)
    hop_len   = int(PROC_RATE * HOP_MS   / 1000)
    framer = Framer(frame_len, hop_len, ch_proc, PROC_RATE)

    # Diagnostics
    diag = Diagnostics(PROC_RATE, ch_proc, hop_len) if RUN_RECORD_DIAGNOSTICS else None
    if RUN_RECORD_DIAGNOSTICS:
        diag.on_open(cfg)

    # Callback for breath segments → push to emit_queue
    def on_segment(res: dict):
        # res: {"EnvProfile", "Duration_ms", "IMF":[...], "t_abs_start": seconds}
        if RUN_TRANSFORM_DIAGNOSTICS:
            print(f"[BREATH] t={res['t_abs_start']:.3f}s, "
                  f"dur={res['Duration_ms']:.1f} ms, env={res['EnvProfile']}, "
                  f"IMF1={res['IMF'][0]:.4f}")

        if emit_queue is not None:
            try:
                emit_queue.put_nowait(res)
            except queue.Full:
                # drop oldest to keep moving
                try:
                    _ = emit_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    emit_queue.put_nowait(res)
                except queue.Full:
                    pass

    # Instantiate the detector once
    RTDetector = RealTimeBreathDetector(PROC_RATE, on_segment=on_segment)

    # Main processing loop
    while not stop_evt.is_set():  # keeps processing until interrupted
        try:
            # pull the next audio block from the queue
            block_f32, t_block_start = blocks.get(timeout=0.5)
        except queue.Empty:
            continue

        # resample captured audio into the processing rate
        block_proc = _resample_poly(block_f32, rate_in, PROC_RATE)

        # continue if the input is tiny/degenerate
        if block_proc.size == 0:
            continue

        # slice the audio block into overlapping analysis frames and feed them
        for frame_f32, t_frame_start in framer.push(block_proc, t_block_start):
            if RUN_RECORD_DIAGNOSTICS and diag is not None:
                # check for DC offset, per-channel RMS, dominant freq, correlation, etc.
                diag.check_frame(frame_f32, t_frame_start)

            # push frames into RTDetector which analyzes and emits HHT/IMFs via on_segment
            RTDetector.push(frame_f32, t_frame_start)