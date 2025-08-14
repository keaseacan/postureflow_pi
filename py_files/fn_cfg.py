# ---- Requested capture & processing settings ----
REQ_RATE = 16000        # recording sampling rate
REQ_CH   = 2              # request 2 channels (ReSpeaker 2-mic HAT)
PERIOD = int(REQ_RATE * 0.016) # 16 ms period

# We'll always process at this rate; channel count follows the input (stereo).
PROC_RATE = 16000         # downstream DSP sample rate (fixed)

# ---- Analysis frame settings ----
FRAME_MS = 25             # analysis window length (e.g., 25 ms for speech)
HOP_MS   = 10             # hop between frames (10 ms → 60% overlap at 25 ms)

# ---- ML Files ----
SVM_MODEL_PATH = "svm_model_final.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

# ---- Diagnostics (enable/disable) ----
# Slow, takes up thread
RUN_RECORD_DIAGNOSTICS = False 
RUN_MORE_RECORD_DIAGNOSTICS = False
RUN_TRANSFORM_DIAGNOSTICS = True
RUN_CLASSIFICATION_DIAGNOSTICS = True
RUN_JSON_DIAGNOSTICS = True
RUN_CORE_DIAGNOSTICS = True
RUN_LIVE_BLE_CLASSIFIERS = True

# ---- BLE Configs ----# 
DEVICE_NAME = "PosturePi"
GPIO_BUTTON_PIN = 17      # BCM (pin 11)
BUTTON_HOLD_SEC = 1.2

DEMO_HEARTBEAT = True           # <<< enable heartbeat
DEMO_HEARTBEAT_SEC = 2          # <<< heartbeat interval

# NUS UUIDs
NUS_SERVICE_UUID = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E'
NUS_RX_UUID      = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E'
NUS_TX_UUID      = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E'

# --- ADV privacy toggle ---
ADVERTISE_SERVICE_UUIDS = True