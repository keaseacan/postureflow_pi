import alsaaudio
import numpy as np

def _fmt_name(fmt):
    return {
        getattr(alsaaudio, 'PCM_FORMAT_S16_LE', None): 'S16_LE',
        getattr(alsaaudio, 'PCM_FORMAT_S24_LE', None): 'S24_LE(24-in-32)',
        getattr(alsaaudio, 'PCM_FORMAT_S32_LE', None): 'S32_LE',
    }.get(fmt, str(fmt))

class Diagnostics:
    """
    Lightweight online checks for: range, DC, clipping, hop timing,
    L/R correlation, dominant freq per channel, rough bit-depth heuristic.
    """
    def __init__(self, sr_proc, ch, hop_len_samples):
        self.sr = sr_proc
        self.ch = ch
        self.hop = hop_len_samples
        self.last_frame_t = None
        self.acc = []   # accumulate ~1s of audio for FFT checks

    def on_open(self, cfg):
        print(f"[DIAG] Opened: dev={cfg['device']} rate_in={cfg['rate_in']} ch_in={cfg['ch_in']} fmt={_fmt_name(cfg['fmt'])}")
        print(f"[DIAG] Processing at {self.sr} Hz, hop={self.hop} samples ({self.hop/self.sr*1000:.1f} ms), channels={self.ch}")

    def check_frame(self, frame_f32, t_frame_start):
        # Basic stats
        mx = float(np.max(np.abs(frame_f32)))
        mean = float(frame_f32.mean())
        clip_ratio = float((np.abs(frame_f32) >= 0.999).mean())

        if clip_ratio > 0.01:
            print(f"[WARN] Clipping {clip_ratio*100:.1f}% @ {t_frame_start:.3f}s (max={mx:.3f})")
        if abs(mean) > 1e-3:
            print(f"[WARN] DC offset ~{mean:.4f} @ {t_frame_start:.3f}s")

        # Frame timestamp hop check
        if self.last_frame_t is not None:
            expected = self.hop / self.sr
            dt = t_frame_start - self.last_frame_t
            if abs(dt - expected) > 0.002:  # >2 ms drift
                print(f"[WARN] Frame hop {dt*1000:.1f} ms (expected {expected*1000:.1f} ms) @ {t_frame_start:.3f}s")
        self.last_frame_t = t_frame_start

        # Accumulate ~1 second for FFT/channel checks
        self.acc.append(frame_f32)
        n_have = sum(chunk.shape[0] for chunk in self.acc)
        if n_have >= self.sr:
            buf = np.vstack(self.acc)[:self.sr, :]  # ~1 s, shape [sr, ch]
            self.acc.clear()
            self._fft_channel_checks(buf, t_frame_start)

    def _fft_channel_checks(self, x, t_ref):
        ch = x.shape[1]
        # RMS per channel
        rms = [float(np.sqrt(np.mean(x[:, i]**2))) for i in range(ch)]
        if ch == 2:
            corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
            if abs(corr) > 0.98:
                print(f"[WARN] L/R highly correlated (corr={corr:.3f}) @ {t_ref:.3f}s; check wiring/routing.")
        print(f"[DIAG] ~1s RMS per ch: {['%.4f' % v for v in rms]} @ {t_ref:.2f}s")

        # Dominant frequency per channel (useful with tone test)
        n = x.shape[0]
        win = np.hanning(n).astype(np.float32)
        freqs = []
        for i in range(ch):
            X = np.fft.rfft(x[:, i] * win)
            k = np.argmax(np.abs(X)[1:]) + 1
            freqs.append(k * self.sr / n)
        print(f"[DIAG] Dominant freq per ch: {['%.1fHz' % f for f in freqs]}")

        # Rough bit-depth guess (heuristic)
        def residual_for_bits(sig, B):
            scale = (2**(B-1) - 1)
            q = np.round(sig * scale) / scale
            return float(np.mean((sig - q) ** 2))
        i_min = int(np.argmin(rms))
        r16 = residual_for_bits(x[:, i_min], 16)
        r24 = residual_for_bits(x[:, i_min], 24)
        guess = 24 if r24 < r16 * 0.6 else 16
        print(f"[DIAG] Bit-depth heuristic: r16={r16:.2e}, r24={r24:.2e} → ~{guess}-bit")
