import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            prediction INTEGER,
            confidence REAL,
            status TEXT,
            fraud_alert INTEGER
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(input_data, prediction, confidence, status, fraud_alert):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_logs
        (timestamp, input_data, prediction, confidence, status, fraud_alert)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(datetime.now()),
        str(input_data),
        prediction,
        confidence,
        status,
        fraud_alert
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, prediction, confidence, status, fraud_alert
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows