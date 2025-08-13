# dependencies
import sqlite3, json, time, uuid, threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Sequence

_DEFAULT_DB = "posture_spool.db"

# timestamp
def _now_ms() -> int:
	return int(time.time() * 1000)

# create SQL table if it doesn't exist
def _ensure_schema(conn: sqlite3.Connection) -> None:
	conn.execute("""
	CREATE TABLE IF NOT EXISTS spool (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts_ms INTEGER NOT NULL,
			cls_idx INTEGER NOT NULL,
			cls_label TEXT NOT NULL,
			score REAL,
			extras TEXT,                           -- JSON blob (nullable)
			state TEXT NOT NULL DEFAULT 'queued',  -- queued | inflight | awaiting_ack
			attempts INTEGER NOT NULL DEFAULT 0,
			next_attempt_ms INTEGER NOT NULL DEFAULT 0,
			ack_deadline_ms INTEGER NOT NULL DEFAULT 0,  -- re-send if no ACK by this time
			lock_uuid TEXT
	);
	""")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_spool_state_ts ON spool(state, ts_ms);")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_spool_next_attempt ON spool(next_attempt_ms);")
	
@dataclass
class ClassEvent:
	ts_ms: int
	cls_idx: int
	cls_label: str
	score: float
	meta: Optional[Dict[str, Any]] = None

# SQL table holder object
class Spool:
	"""Durable outbox (SQLite) for classification change-events."""
	def __init__(self, path: str = _DEFAULT_DB):
		self.conn = sqlite3.connect(path, check_same_thread=False)
		self.conn.row_factory = sqlite3.Row
		# Low-latency settings
		self.conn.execute("PRAGMA journal_mode=WAL;")
		self.conn.execute("PRAGMA synchronous=NORMAL;")
		self.conn.execute("PRAGMA temp_store=MEMORY;")
		_ensure_schema(self.conn)
		self._lock = threading.Lock()

	# push to SQL table
	def enqueue(self, ev: ClassEvent) -> int:
		with self._lock, self.conn:
			cur = self.conn.execute(
				"""INSERT INTO spool(ts_ms, cls_idx, cls_label, score, extras, state, attempts, next_attempt_ms)
				   VALUES(?,?,?,?,?,'queued',0,0);""",
				(ev.ts_ms, ev.cls_idx, ev.cls_label, ev.score,
				 json.dumps(ev.meta, separators=(",", ":")) if ev.meta else None)
			)
			return cur.lastrowid

	# re-queue expired 'inflight' rows whose lease has timed out
	def _reclaim_expired(self) -> None:
		now = _now_ms()
		with self._lock, self.conn:
			# inflight lease expired -> back to queued
			self.conn.execute(
				"UPDATE spool SET state='queued', lock_uuid=NULL "
				"WHERE state='inflight' AND next_attempt_ms <= ?;",
				(now,)
			)
			# awaiting_ack timeout -> re-send
			self.conn.execute(
				"UPDATE spool SET state='queued', attempts=attempts+1, next_attempt_ms=? "
				"WHERE state='awaiting_ack' AND ack_deadline_ms > 0 AND ack_deadline_ms <= ?;",
				(now + 2000, now)  # small delay before retry
			)

	# return rows that are ready to be sent
	def claim_batch(self, limit: int = 32, lease_ms: int = 10000):
		"""Lease up to `limit` queued rows (oldest first). Returns (rows, token).
		Also sets state='inflight', records a lease token, and pushes next_attempt_ms forward by lease_ms.
		"""
		now = _now_ms()
		token = str(uuid.uuid4())

		# first, reclaim any inflight rows whose lease expired
		self._reclaim_expired()

		with self._lock, self.conn:
			rows = self.conn.execute(
				"""SELECT id FROM spool
				   WHERE state='queued' AND next_attempt_ms <= ?
				   ORDER BY ts_ms ASC
				   LIMIT ?;""",
				(now, limit)
			).fetchall()
			if not rows:
				return [], token
			ids = [r["id"] for r in rows]
			q = ",".join("?" for _ in ids)
			self.conn.execute(
				f"""UPDATE spool SET state='inflight', lock_uuid=?, next_attempt_ms=?
					WHERE id IN ({q});""",
				(token, now + lease_ms, *ids)
			)
			full = self.conn.execute(
				f"""SELECT * FROM spool WHERE id IN ({q});""",
				(*ids,)
			).fetchall()
			return full, token

	# mark in the SQL table that the row has been sent
	def mark_sent(self, ids: Sequence[int]) -> None:
		if not ids: return
		q = ",".join("?" for _ in ids)
		with self._lock, self.conn:
			self.conn.execute(f"DELETE FROM spool WHERE id IN ({q});", (*ids,))

	# mark in the SQL table that the row failed to send
	def mark_failed(self, ids: Sequence[int]) -> None:
		if not ids: return
		now = _now_ms()
		with self._lock, self.conn:
			for i in ids:
				row = self.conn.execute("SELECT attempts FROM spool WHERE id=?;", (i,)).fetchone()
				attempts = (row[0] if row else 0) + 1
				delay = min(60000, 1000 * (2 ** (attempts - 1)))  # 1s,2s,4s... <=60s
				self.conn.execute(
					"UPDATE spool SET state='queued', attempts=?, next_attempt_ms=? WHERE id=?;",
					(attempts, now + delay, i)
				)
	# mark as delivered; wait for phone ACK before delete
	def mark_delivered(self, ids: Sequence[int], ack_timeout_ms: int = 30000) -> None:
		if not ids: return
		now = _now_ms()
		q = ",".join("?" for _ in ids)
		with self._lock, self.conn:
			self.conn.execute(
					f"UPDATE spool SET state='awaiting_ack', ack_deadline_ms=? WHERE id IN ({q});",
					(now + ack_timeout_ms, *ids)
			)

	# phone ACK handler can call this to delete
	def ack(self, ids: Sequence[int]) -> None:
		if not ids: return
		q = ",".join("?" for _ in ids)
		with self._lock, self.conn:
			self.conn.execute(f"DELETE FROM spool WHERE id IN ({q});", (*ids,))

	# keep at max_rows when ccalled, prunes the rest
	def prune_keep_most_recent(self, max_rows: int = 100000) -> None:
		with self._lock, self.conn:
			# count only queued rows to avoid deleting inflight/awaiting_ack
			(count,) = self.conn.execute("SELECT COUNT(*) FROM spool WHERE state='queued';").fetchone()
			extra = count - max_rows
			if extra > 0:
				self.conn.execute(
					"DELETE FROM spool WHERE id IN ("
					"  SELECT id FROM spool WHERE state='queued' ORDER BY ts_ms ASC LIMIT ?"
					");",
					(extra,)
				)

	def wal_checkpoint(self) -> None:
		with self._lock, self.conn:
			self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

	# close SQL table instance
	def close(self) -> None:
		with self._lock:
			self.conn.close()

class Transport:
	"""Abstract transport interface."""
	def send_batch(self, batch_events: List[Dict[str, Any]]) -> bool:
		raise NotImplementedError

class StdoutTransport(Transport):
	"""Debug transport: dumps full DB rows as JSON (not used for wire)."""
	def send_batch(self, batch_events: List[Dict[str, Any]]) -> bool:
		print(json.dumps({"type": "debug_dump", "n": len(batch_events), "rows": batch_events},
						 separators=(",", ":")))
		return True

class SpoolWorker(threading.Thread):
	def __init__(self, spool, transport,
                 batch_size: int = 32,
                 idle_sleep_s: float = 0.2,
                 prune_interval_s: int = 900,      # every 15 min
                 prune_max_rows: int = 100_000,
                 checkpoint_interval_s: int = 900  # every 15 min
                 ):
		super().__init__(daemon=True)
		self.spool = spool
		self.transport = transport
		self.batch_size = batch_size
		self.idle_sleep_s = idle_sleep_s
		self.prune_interval_s = prune_interval_s
		self.prune_max_rows = prune_max_rows
		self.checkpoint_interval_s = checkpoint_interval_s
		self._stop = threading.Event()
		self._last_prune_ms = 0
		self._last_ckpt_ms = 0

	def stop(self) -> None:
		self._stop.set()

	def run(self) -> None:
		while not self._stop.is_set():
			now = _now_ms()

			# periodic prune (cheap seatbelt)
			if now - self._last_prune_ms >= self.prune_interval_s * 1000:
				self.spool.prune_keep_most_recent(self.prune_max_rows)
				self._last_prune_ms = now

			rows, _token = self.spool.claim_batch(limit=self.batch_size)

			if not rows:
				# idle: nice time to checkpoint occasionally
				if now - self._last_ckpt_ms >= self.checkpoint_interval_s * 1000:
					self.spool.wal_checkpoint()
					self._last_ckpt_ms = now
				time.sleep(self.idle_sleep_s)
				continue

		# normalize DB rows -> event dicts for the transport
		events = []
		ids = []
		for r in rows:
			ids.append(r["id"])
			events.append({
				"id": r["id"],
				"ts_ms": r["ts_ms"],
				"cls": {"idx": r["cls_idx"], "label": r["cls_label"], "score": r["score"]},
				"meta": json.loads(r["extras"]) if r["extras"] else None
			})

		# send the batch (transport returns True/False)
		ok = False
		try:
			ok = self.transport.send_batch(events)
		except Exception:
			ok = False

		if ok:
			# ACK mode: mark as delivered and wait for phone ACK to delete later
			# (requires Spool.mark_delivered and Spool.ack implemented)
			self.spool.mark_delivered(ids, ack_timeout_ms=30000)  # 30s re-send if no ACK
		else:
			# send failed or not ready (e.g., no BLE subscriber) -> backoff and retry later
			self.spool.mark_failed(ids)
