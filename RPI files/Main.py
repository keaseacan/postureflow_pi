import Time_Fun as ttf
import BT_Fun as btf

# ---- Functions ----
def BT_setup():
    """Sets up Bluetooth connection."""
    print("Setting up Bluetooth...")

def IMU_setup():
    """Sets up IMU sensor."""
    print("Setting up IMU...")

# ---- Setup ----
def setup():
    """Runs once at the start."""
    print("Setup: initializing hardware...")
    ttf.set_time()


def loop():
    """Runs repeatedly forever."""
    print("Loop: main code here")
    time.sleep(1)  # 1 second delay (like delay(1000) in Arduino)

# ---- Main program ----
if __name__ == "__main__":
    setup()
    while True:
        loop()