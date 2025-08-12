# functions
from py_files.fn_time import setup_i2c, write_to_pi
from py_files.record_process.fn_record_main import stop_audio_pipeline, _feature_q as feat_q
from py_files.fn_model import classify_imfs

# dependencies
import queue
import time

# ---- Setup ----
def pi_setup():
	"""Runs once at the start."""
	print("Setup: initializing hardware...")
	setup_i2c()
	write_to_pi()

def pi_loop():
	"""Runs repeatedly forever until interrupted"""
	try:
		res = feat_q.get_nowait()
		feats = res["IMF"]
		output = classify_imfs(imfs=feats)
		print(output)
	except queue.Empty:
		pass
	time.sleep(0.01)


pi_setup()
while True:
  pi_loop()
stop_audio_pipeline()
