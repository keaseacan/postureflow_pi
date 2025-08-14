# ---------- BCD helpers ----------
def bcd_to_int(b: int) -> int: return (b >> 4) * 10 + (b & 0x0F)
def int_to_bcd(d: int) -> int: return ((d // 10) << 4) | (d % 10)