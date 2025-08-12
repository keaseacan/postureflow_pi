import numpy as np

class Framer:
  """
  Maintains a small float32 buffer and emits overlapping frames of length
  frame_len with hop hop_len. Tracks a start-time for the buffer so we
  can timestamp each emitted frame accurately. Works for multi-channel.
  """
  def __init__(self, frame_len, hop_len, channels, sr):
    self.frame_len = frame_len
    self.hop_len   = hop_len
    self.ch        = channels
    self.sr        = sr
    self.buf = np.empty((0, channels), dtype=np.float32)  # rolling buffer
    self.buf_t0 = None  # absolute start time (seconds, monotonic-based) of buf[0]

  def push(self, block_f32, t_block_start):
    """
    Append a new block [N, ch] that begins at absolute time t_block_start,
    then emit as many frames as possible.
    Returns: list of (frame[frame_len, ch], t_frame_start_seconds).
    """
    out = []
    if self.buf.size == 0:
      # First data after empty state: anchor buffer time to this block's start.
      self.buf_t0 = t_block_start

    # Append new samples to end of buffer
    if self.buf.shape[0] == 0:
      self.buf = block_f32
    else:
      self.buf = np.vstack((self.buf, block_f32))  # (copy) simple & clear

    # While enough samples exist, emit a frame and slide by hop
    while self.buf.shape[0] >= self.frame_len:
      frame = self.buf[:self.frame_len]              # [frame_len, ch]
      t_frame_start = self.buf_t0                    # absolute time for frame[0]
      out.append((frame.copy(), t_frame_start))      # copy to decouple from buffer

      # Slide window forward by hop_len samples and advance time base
      self.buf = self.buf[self.hop_len:]
      self.buf_t0 += self.hop_len / self.sr

    return out
  