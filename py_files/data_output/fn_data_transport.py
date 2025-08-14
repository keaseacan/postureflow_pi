# dependencies
import json
from typing import List, Dict, Any

# functions
from py_files.bt.bt_transport import ble_send

# constants
from py_files.fn_cfg import RUN_JSON_DIAGNOSTICS

class ChangeEventTransport:
  """
  Converts SpoolWorker's normalized rows to minimal 'v1.change' JSON and prints it.
  Swap this later with a BLE transport that notifies the same payload.
  """
  def __init__(self, include_label: bool = False):
    self.include_label = include_label

  def send_batch(self, batch_events: List[Dict[str, Any]]) -> bool:
    wire = []
    for e in batch_events:
      item = {
        "id": e["id"],         # spool row id (optional ACK later)
        "t":  e["ts_ms"],      # start time of segment
        "i":  e["cls"]["idx"], # label index
      }
      if self.include_label and e["cls"].get("label"):
        item["l"] = e["cls"]["label"]
      wire.append(item)

    payload = {
      "type": "posture_batch",
      "schema": "v1.change",
      "n": len(wire),
      "events": wire
    }
    if RUN_JSON_DIAGNOSTICS:
      print(json.dumps(payload, separators=(",", ":")))
    ble_send(payload)
    return True
