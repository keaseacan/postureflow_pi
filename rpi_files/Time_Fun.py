import subprocess
from datetime import datetime
import smbus2 as sm
from BT_Fun import cts_char, ble_mac

# definitions
i2c_port: int = 1
rtc_addr: hex = 0x68

# helper functions
def bcd_to_int(b): return (b >> 4) * 10 + (b & 0x0F)
def int_to_bcd(d): return ((d // 10) << 4) | (d % 10)

# for checking on rpi init
def setup_i2c()->bool:
  try:
    with sm.SMBus(i2c_port) as _:  # run for side effects
      print("i2c init successful")
      return True
  except Exception:
    return False
  
def read_rtc()->list:
  try:
    with sm.SMBus(i2c_port) as _:
      td = _.read_i2c_block_data(rtc_addr, 0x00, 7)
      print("ds3231 rtc successfuly read")
      return td
  except Exception:
    print("ds3231 rtc not read")
  
def write_to_pi()->bool:
  try:
    s, m, h, dw, d, mo, y = read_rtc()
    dt: datetime = datetime(2000 + bcd_to_int(y),
                bcd_to_int(mo & 0x1F),
                bcd_to_int(d),
                bcd_to_int(h & 0x3F),
                bcd_to_int(m),
                bcd_to_int(s & 0x7F))
    subprocess.run(["sudo", "date", "-s", dt.strftime("%Y-%m-%d %H:%M:%S")])
    print("pi clock successfully overwritten")
    return True
  except Exception:
    print("pi clock not overwritten")
    return False

"""
def write_to_rtc()->bool:
  try:
    now: datetime = read_ble_time()
    data = [
      int_to_bcd(now.second),
      int_to_bcd(now.minute),
      int_to_bcd(now.hour),
      int_to_bcd((now.isoweekday()) % 7 or 7),  # 1=Mon..7=Sun
      int_to_bcd(now.day),
      int_to_bcd(now.month),
      int_to_bcd(now.year - 2000)
    ]
    try:
      with sm.SMBus(i2c_port) as _:
      _.write_i2c_block_data(rtc_addr, 0x00, data)
    except Exception:
      return False 
  except Exception:
    return False

async def read_ble_time()->datetime:
  async with BleakClient(ble_mac) as client:
    if not client.is_connected:
      raise RuntimeError("Failed to connect to BLE device.")
  data = await client.read_gatt_char(cts_char)
  return parse_cts(bytes(data))


def parse_cts(payload: bytes)->datetime:
    # CTS format (Little-Endian):
    # year u16, month u8, day u8, hour u8, min u8, sec u8, dow u8, frac256 u8, adj_reason u8
    year = int.from_bytes(payload[0:2], "little")
    month = payload[2]
    day = payload[3]
    hour = payload[4]
    minute = payload[5]
    second = payload[6]
    return datetime(year, month, day, hour, minute, second)
"""