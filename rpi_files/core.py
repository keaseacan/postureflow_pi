import fn_time as ttf

# ---- Setup ----
def pi_setup():
	"""Runs once at the start."""
	print("Setup: initializing hardware...")
	ttf.setup_i2c()
	ttf.write_to_pi()

def pi_loop():
	"""Runs repeatedly forever."""
	print("Loop: main code here")

pi_setup()