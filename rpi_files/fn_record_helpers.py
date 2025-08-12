import alsaaudio
import numpy as np

def _resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """
    Minimal linear resampler supporting 1D [N] or 2D [N, ch].
    Good enough for 48k→16k telemetry. For perfect continuity, keep state/FIR.
    """
    if in_sr == out_sr or x.size == 0:
        return x.astype(np.float32, copy=False)

    n_in = x.shape[0]
    n_out = int(round(n_in * (out_sr / float(in_sr))))
    if n_out <= 1:
        # Return an empty array with matching dimensionality
        if x.ndim == 1:
            return np.empty((0,), dtype=np.float32)
        else:
            return np.empty((0, x.shape[1]), dtype=np.float32)

    t_in  = np.linspace(0.0, 1.0, num=n_in,  endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)

    if x.ndim == 1:
        y = np.interp(t_out, t_in, x).astype(np.float32, copy=False)
    else:
        ch = x.shape[1]
        y = np.empty((n_out, ch), dtype=np.float32)
        for c in range(ch):
            y[:, c] = np.interp(t_out, t_in, x[:, c])
    return y

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
