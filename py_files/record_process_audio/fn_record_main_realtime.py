import queue
import threading
import time
from typing import Optional, Callable

import alsaaudio
import numpy as np

from py_files.record_process_audio.fn_record_helpers import cfg, _bytes_to_float32, _resample_poly
from py_files.record_process_audio.fn_record_deviceinit import open_pcm_with_fallback
from py_files.record_process_audio.fn_record_buffer import stop_evt, Framer
from py_files.record_process_audio.fn_record_debug import Diagnostics
from py_files.record_process_audio.fn_process_breath import RealTimeBreathDetector

from py_files.fn_cfg import PERIOD, PROC_RATE, FRAME_MS, HOP_MS
from py_files.fn_cfg import RUN_RECORD_DIAGNOSTICS, RUN_TRANSFORM_DIAGNOSTICS

_thread: Optional[threading.Thread] = None
_result_q: Optional[queue.Queue] = None
_state_lock = threading.Lock()


def start_audio_pipeline(
    on_segment: Optional[Callable[[dict], None]] = None,
    out_queue: Optional[queue.Queue] = None,
) -> queue.Queue:
    """
    Start the realtime pipeline.

    If on_segment is provided, each completed breath segment is sent directly to
    that callback in the SAME audio worker thread. This avoids the extra
    classifier thread and queue latency.

    If out_queue is not provided, an internal queue is created and returned.
    """
    global _thread, _result_q

    with _state_lock:
        if _thread is not None and _thread.is_alive():
            raise RuntimeError("Audio pipeline already running")

        stop_evt.clear()
        _result_q = out_queue or queue.Queue(maxsize=64)

        def worker():
            pcm = None
            try:
                pcm, rate_in, ch_in, dev, fmt = open_pcm_with_fallback()
                cfg['rate_in'], cfg['ch_in'], cfg['device'], cfg['fmt'] = rate_in, ch_in, dev, fmt

                frame_len = int(PROC_RATE * FRAME_MS / 1000)
                hop_len   = int(PROC_RATE * HOP_MS   / 1000)
                framer = Framer(frame_len, hop_len, ch_in, PROC_RATE)

                diag = Diagnostics(PROC_RATE, ch_in, hop_len) if RUN_RECORD_DIAGNOSTICS else None
                if RUN_RECORD_DIAGNOSTICS and diag is not None:
                    diag.on_open(cfg)

                # Capture timing reference
                samples_captured = 0
                t0 = time.monotonic()

                def emit_result(res: dict):
                    if RUN_TRANSFORM_DIAGNOSTICS:
                        print(
                            f"[BREATH] t={res['t_abs_start']:.3f}s, "
                            f"dur={res['Duration_ms']:.1f} ms, env={res['EnvProfile']}, "
                            f"IMF1={res['IMF'][0]:.4f}",
                            flush=True,
                        )

                    if on_segment is not None:
                        on_segment(res)

                    if _result_q is not None:
                        try:
                            _result_q.put_nowait(res)
                        except queue.Full:
                            try:
                                _result_q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                _result_q.put_nowait(res)
                            except queue.Full:
                                pass

                detector = RealTimeBreathDetector(PROC_RATE, on_segment=emit_result)
                # Faster demo cadence than the stock defaults
                detector.max_buffer_sec = 3.0
                detector.tail_guard_sec = 0.05
                detector.min_analyze_sec = 0.15

                # Blocking capture loop: one owner for ALSA, stable timing.
                while not stop_evt.is_set():
                    try:
                        nframes, data = pcm.read()
                    except alsaaudio.ALSAAudioError as e:
                        # Rare recoverable read hiccup: nap very briefly.
                        print(f"[ALSA] read error: {e}", flush=True)
                        time.sleep(0.002)
                        continue

                    if nframes <= 0 or not data:
                        # In PCM_NORMAL this is uncommon, but harmless.
                        time.sleep(0.001)
                        continue

                    block_f32 = _bytes_to_float32(data, fmt, ch_in)
                    if block_f32.size == 0:
                        continue

                    t_block_start = samples_captured / rate_in
                    samples_captured += block_f32.shape[0]

                    block_proc = _resample_poly(block_f32, rate_in, PROC_RATE)
                    if block_proc.size == 0:
                        continue

                    for frame_f32, t_frame_start in framer.push(block_proc, t0 + t_block_start):
                        if RUN_RECORD_DIAGNOSTICS and diag is not None:
                            diag.check_frame(frame_f32, t_frame_start)
                        detector.push(frame_f32, t_frame_start)

            except Exception as e:
                print(f"[PIPE] worker fatal: {e}", flush=True)
                raise
            finally:
                try:
                    if pcm is not None:
                        pcm.close()
                except Exception:
                    pass

        _thread = threading.Thread(target=worker, daemon=True, name="audio-realtime")
        _thread.start()
        return _result_q


def stop_audio_pipeline(timeout: float = 1.5):
    global _thread
    stop_evt.set()

    with _state_lock:
        t = _thread
        _thread = None

    if t is not None and t.is_alive():
        try:
            t.join(timeout=timeout)
        except RuntimeError:
            pass
'''

fn_classification_direct = r'''# Direct classification helpers for demo mode.
# No classification thread. Classify immediately when a segment is emitted.

import time
from typing import Callable, Optional

import numpy as np

from py_files.model.fn_classification_model import classify_imfs, class_map


def classify_result(res: dict) -> dict:
    """
    Convert a detector result dict into a classifier output dict.
    Expected input:
      {"EnvProfile", "Duration_ms", "IMF", "t_abs_start"}
    """
    imfs = np.asarray(res["IMF"], dtype=np.float32)
    idx, label, margin = classify_imfs(imfs)

    try:
        t_wall = time.time()
    except Exception:
        t_wall = 0.0

    out = {
        "idx": int(idx) if idx is not None else -1,
        "label": str(label),
        "dur_ms": float(res.get("Duration_ms", 0.0)),
        "env": res.get("EnvProfile", "unknown"),
        "t": t_wall,
        "margin": margin,
        "IMF": res["IMF"],
    }
    return out


def print_classification(out: dict):
    print(
        f"State: {out['label']}, "
        f"Duration: {out['dur_ms']:.1f} ms, "
        f"Env: {out['env']}",
        flush=True,
    )


def make_segment_callback(
    handler: Optional[Callable[[dict], None]] = None,
) -> Callable[[dict], None]:
    """
    Returns a callback suitable for start_audio_pipeline(on_segment=...)
    """
    if handler is None:
        handler = print_classification

    def _cb(res: dict):
        out = classify_result(res)
        handler(out)

    return _cb