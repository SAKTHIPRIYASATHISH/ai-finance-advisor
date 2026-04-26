import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'finance.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            payment_mode TEXT NOT NULL,
            is_anomaly INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def add_transaction(date, description, amount, category, payment_mode):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions (date, description, amount, category, payment_mode)
        VALUES (?, ?, ?, ?, ?)
    """, (date, description, float(amount), category, payment_mode))
    conn.commit()
    conn.close()


def get_all_transactions():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY date DESC", conn
    )
    conn.close()
    return df


def get_by_month(year, month):
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM transactions
        WHERE strftime('%Y', date) = ?
        AND strftime('%m', date) = ?
        ORDER BY date DESC
    """, conn, params=(str(year), f"{month:02d}"))
    conn.close()
    return df


def get_monthly_totals():
    """Returns month-wise total spending — used by the predictor."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m', date) as month,
            SUM(amount) as total,
            COUNT(*) as count
        FROM transactions
        GROUP BY month
        ORDER BY month
    """, conn)
    conn.close()
    return df


def get_category_monthly(category):
    """Returns monthly totals for a specific category."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            strftime('%Y-%m', date) as month,
            SUM(amount) as total
        FROM transactions
        WHERE category = ?
        GROUP BY month
        ORDER BY month
    """, conn, params=(category,))
    conn.close()
    return df


def seed_from_csv(csv_path):
    """Load synthetic data into DB — call only once on first run."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]
    if count > 0:
        conn.close()
        return
    df = pd.read_csv(csv_path)
    df.to_sql('transactions', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f"Seeded {len(df)} transactions.")


def delete_transaction(transaction_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()


def add_user(username, name, hashed_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, name, password) VALUES (?, ?, ?)",
        (username, name, hashed_password)
    )
    conn.commit()
    conn.close()


def get_user(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, name TEXT NOT NULL, password TEXT NOT NULL)")
    conn.commit()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, name TEXT NOT NULL, password TEXT NOT NULL)")
    conn.commit()
    cursor.execute("SELECT username, name, password FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_anomaly(transaction_id, flag=1):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE transactions SET is_anomaly = ? WHERE id = ?",
        (flag, transaction_id)
    )
    conn.commit()
    conn.close()