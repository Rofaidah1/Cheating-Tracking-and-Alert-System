import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "alerts" / "cheating_alerts.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS cheating_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                behavior TEXT NOT NULL,
                confidence REAL NOT NULL,
                screenshot_path TEXT NOT NULL,
                student_track_id INTEGER
            )
            '''
        )
        conn.commit()


def insert_alert(behavior, confidence, screenshot_path, student_track_id=None):
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            '''
            INSERT INTO cheating_alerts
            (event_time, behavior, confidence, screenshot_path, student_track_id)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (event_time, behavior, float(confidence), str(screenshot_path), student_track_id),
        )
        conn.commit()
        return cur.lastrowid


def get_alerts():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            '''
            SELECT id, event_time, behavior, confidence,
                   screenshot_path, student_track_id
            FROM cheating_alerts
            ORDER BY id DESC
            '''
        ).fetchall()
    return [dict(row) for row in rows]


init_db()
