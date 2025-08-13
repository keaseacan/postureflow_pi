import time
from py_files.record_process_audio.fn_record_main import start_audio_pipeline, stop_audio_pipeline
from py_files.model.fn_classification_main import start_classification, stop_classification
from py_files.fn_time import setup_i2c, write_to_pi

def pi_setup():
    print("Setup: initializing hardware...")
    setup_i2c()
    write_to_pi()

		# get IMF queue
    feat_q = start_audio_pipeline()

		#  start classifier thread
    start_classification(feat_q)
    return feat_q

if __name__ == "__main__":
    try:
        _ = pi_setup()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_classification()
        stop_audio_pipeline()