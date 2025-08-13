# import functions
from py_files.record_process_audio.fn_record_threads import capture_thread, processing_thread
from py_files.record_process_audio.fn_record_buffer import stop_evt

# dependencies
import queue
import threading
import time
import signal

# Public handles for host app
_feature_q: queue.Queue | None = None
_threads = {}

def start_audio_pipeline(out_queue: queue.Queue | None = None) -> queue.Queue:
	"""
	Start capture + processing threads. Returns the queue that receives
	result dicts from RealTimeBreathDetector.on_segment, e.g.:
		{"EnvProfile": str, "Duration_ms": float, "t_abs_start": float, "IMF": [..]}
	"""
	global _feature_q, _threads
	_feature_q = out_queue or queue.Queue(maxsize=512)
	stop_evt.clear()

	t_cap  = threading.Thread(target=capture_thread,   daemon=True, name="audio-capture")
	t_proc = threading.Thread(target=processing_thread, daemon=True, name="audio-process",
														kwargs={"emit_queue": _feature_q})
	_threads = {"cap": t_cap, "proc": t_proc}
	t_cap.start(); t_proc.start()
	return _feature_q

def stop_audio_pipeline(timeout: float = 1.0):
	"""Signal threads to stop and join them."""
	stop_evt.set()
	for t in _threads.values():
			try:
					t.join(timeout=timeout)
			except RuntimeError:
					pass

def main():
	"""
	Demo runner: installs Ctrl+C handlers, starts the pipeline, and idles.
	In your real app, call start_audio_pipeline() once and poll the returned queue.
	"""
	def on_sigint(sig, frm):
		stop_evt.set()
	try:
		signal.signal(signal.SIGINT, on_sigint)
		signal.signal(signal.SIGTERM, on_sigint)
	except Exception:
		print("Signal setup failed")

	start_audio_pipeline()  # use internal queue
	try:
		while not stop_evt.is_set():
			time.sleep(0.5)
	finally:
		stop_audio_pipeline()

if __name__ == "__main__":
    main()