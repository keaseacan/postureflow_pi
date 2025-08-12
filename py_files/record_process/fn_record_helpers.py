# dependencies
import alsaaudio
import numpy as np
from math import gcd
from scipy.signal import resample_poly

# rate_in = sampling rate, based on REQ_RATE if not what the device allows from FORMAT_CANDIDATES
# ch_in = recording channels
# device = what is chosen from DEVICE_CANDIDATES
# fmt = the chosen format from FORMAT_CANDIDATES
cfg = {'rate_in': None, 'ch_in': None, 'device': None, 'fmt': None}  # filled after device opens

def _resample_poly(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    if x.ndim == 1:
        return resample_poly(x, up, down)  # default window is good
    # stereo/multichannel
    return np.stack([resample_poly(x[:, c], up, down) for c in range(x.shape[1])], axis=1)

def _bytes_to_float32(data: bytes, fmt, ch: int) -> np.ndarray:
    """
    Convert ALSA raw bytes to float32 in [-1, 1], shape [N, ch],
    handling S16_LE, S24_LE (24-in-32), and S32_LE.
    """
    if fmt == alsaaudio.PCM_FORMAT_S16_LE:
        x = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
    elif fmt == alsaaudio.PCM_FORMAT_S32_LE:
        x = np.frombuffer(data, dtype='<i4').astype(np.float32) / 2147483648.0  # 2^31
    elif fmt == alsaaudio.PCM_FORMAT_S24_LE:
        x32 = np.frombuffer(data, dtype='<i4')
        x24 = (x32 >> 8).astype(np.int32)      # sign-extend from 24 to 32 bits
        x = x24.astype(np.float32) / 8388608.0 # 2^23
    else:
        raise ValueError(f"Unsupported PCM format: {fmt}")

    # Drop any trailing partial frame and reshape to [N, ch]
    n_samp = (x.size // ch) * ch
    if n_samp == 0:
        return np.empty((0, ch), dtype=np.float32)
    return x[:n_samp].reshape(-1, ch)
