import asyncio, subprocess
from datetime import datetime
from bleak import BleakClient, BleakScanner
import smbus2

def bcd_to_dec(b): return (b >> 4) * 10 + (b & 0x0F)
s, m, h, wd, d, mo, y = bus.read_i2c_block_data(ADDR, 0x00, 7)
dt = datetime(2000 + bcd_to_dec(y),
              bcd_to_dec(mo & 0x1F),
              bcd_to_dec(d),
              bcd_to_dec(h & 0x3F),
              bcd_to_dec(m),
              bcd_to_dec(s & 0x7F))
subprocess.run(["sudo", "date", "-s", dt.strftime("%Y-%m-%d %H:%M:%S")])
print("System time set to:", dt)

def dec_to_bcd(d): return ((d // 10) << 4) | (d % 10)
now = get_time_from_ble()
data = [
    dec_to_bcd(now.second),
    dec_to_bcd(now.minute),
    dec_to_bcd(now.hour),
    dec_to_bcd((now.isoweekday()) % 7 or 7),  # 1=Mon..7=Sun
    dec_to_bcd(now.day),
    dec_to_bcd(now.month),
    dec_to_bcd(now.year - 2000)
]
bus.write_i2c_block_data(ADDR, 0x00, data)


def parse_cts(payload: bytes) -> datetime:
    # CTS format (Little-Endian):
    # year u16, month u8, day u8, hour u8, min u8, sec u8, dow u8, frac256 u8, adj_reason u8
    year = int.from_bytes(payload[0:2], "little")
    month = payload[2]
    day = payload[3]
    hour = payload[4]
    minute = payload[5]
    second = payload[6]
    return datetime(year, month, day, hour, minute, second)

async def get_time_from_ble() -> datetime:
    async with BleakClient(BLE_MAC) as client:
        if not client.is_connected:
            raise RuntimeError("Failed to connect to BLE device.")
        data = await client.read_gatt_char(CTS_CHAR)
        return parse_cts(bytes(data))
    
def take_time():
    bus = smbus2.SMBus(1)
    ADDR = 0x68
    bcd_to_dec()
    bus.close()

def set_time():
    bus = smbus2.SMBus(1)
    ADDR = 0x68
    dec_to_bcd()
    bus.close()