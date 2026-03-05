import sqlite3
import json
import logging
import os
import psycopg2
from psycopg2 import extras
from datetime import datetime, timezone

logger = logging.getLogger('discord_bot.database')

class CursorContext:
    def __init__(self, cursor):
        self.cursor = cursor
    def __enter__(self):
        return self.cursor
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.cursor, 'close'):
            self.cursor.close()

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_url = os.getenv('DATABASE_URL')
        self.is_postgres = False
        
        if self.db_url:
            # Fix for Railway/Heroku postgres URLs
            if self.db_url.startswith("postgres://"):
                self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            
            # Add SSL requirement if not present (common for Railway Postgres)
            if "?" not in self.db_url:
                self.db_url += "?sslmode=require"
            elif "sslmode=" not in self.db_url:
                self.db_url += "&sslmode=require"
                
            try:
                # Test connection immediately
                test_conn = psycopg2.connect(self.db_url)
                test_conn.close()
                self.is_postgres = True
                logger.info("✅ Database: Successfully connected to PostgreSQL.")
            except Exception as e:
                logger.error(f"❌ Database: PostgreSQL connection failed: {e}")
                logger.warning("⚠️ Database: Falling back to SQLite due to PostgreSQL failure.")
                self.is_postgres = False

        if not self.is_postgres:
            if db_path is None:
                # Check environment variable for Railway/Docker volumes
                env_path = os.getenv('DATABASE_PATH')
                if env_path:
                    self.db_path = env_path
                else:
                    # Get the directory where database.py is located
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    self.db_path = os.path.join(base_dir, 'bot_memory.db')
            else:
                self.db_path = db_path
            
            # Ensure the directory for the database exists
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            logger.info(f"💾 Database: Using SQLite at {self.db_path}")
            
        self.init_db()

    def get_cursor(self, conn):
        return CursorContext(conn.cursor())

    def get_connection(self):
        try:
            if self.is_postgres:
                return psycopg2.connect(self.db_url)
            else:
                return sqlite3.connect(self.db_path)
        except Exception as e:
            logger.error(f"Critical error getting DB connection: {e}")
            raise e

    def get_placeholder(self):
        return "%s" if self.is_postgres else "?"

    def init_db(self):
        """Initialize the database tables."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Helper for table creation
            def create_table(sql):
                if not self.is_postgres:
                    sql = sql.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
                    sql = sql.replace('BIGINT', 'INTEGER')
                cursor.execute(sql)

            # Table for conversation history
            create_table('''
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table for user memory/profiles
            create_table('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    profile_summary TEXT,
                    vibe TEXT,
                    notes TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table for Levels
            create_table('''
                CREATE TABLE IF NOT EXISTS user_levels (
                    guild_id BIGINT,
                    user_id BIGINT,
                    xp BIGINT DEFAULT 0,
                    level INTEGER DEFAULT 0,
                    PRIMARY KEY (guild_id, user_id)
                )
            ''')

            # Migration: Ensure guild_id column exists for existing levels tables
            try:
                cursor.execute('ALTER TABLE user_levels ADD COLUMN IF NOT EXISTS guild_id BIGINT DEFAULT 0')
                # Note: IF NOT EXISTS for column is Postgres specific. For SQLite we handle differently.
                if not self.is_postgres:
                    # SQLite doesn't support IF NOT EXISTS in ALTER TABLE
                    # But we can try-except
                    try: cursor.execute('ALTER TABLE user_levels ADD COLUMN guild_id BIGINT DEFAULT 0')
                    except: pass
                conn.commit()
            except Exception as e:
                pass

            # Table for Warnings
            create_table('''
                CREATE TABLE IF NOT EXISTS user_warnings (
                    user_id BIGINT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    history TEXT -- JSON string
                )
            ''')

            # Table for YT Cooldowns
            create_table('''
                CREATE TABLE IF NOT EXISTS yt_cooldowns (
                    user_id BIGINT PRIMARY KEY,
                    expiry TIMESTAMP
                )
            ''')

            # Table for Guild Inviters
            create_table('''
                CREATE TABLE IF NOT EXISTS guild_inviters (
                    guild_id TEXT PRIMARY KEY,
                    user_id BIGINT
                )
            ''')

            # Table for Portfolios
            create_table('''
                CREATE TABLE IF NOT EXISTS user_portfolios (
                    user_id BIGINT PRIMARY KEY,
                    portfolio_data TEXT, -- Changed from portfolio_url to match code
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table for Active Captchas (Missing in previous version)
            create_table('''
                CREATE TABLE IF NOT EXISTS active_captchas (
                    user_id BIGINT PRIMARY KEY,
                    code TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # --- THE SECOND BRAIN (Personal Knowledge Store) ---
            pk_type = "SERIAL" if self.is_postgres else "INTEGER"
            create_table(f'''
                CREATE TABLE IF NOT EXISTS user_brain (
                    id {pk_type} PRIMARY KEY{' AUTOINCREMENT' if not self.is_postgres else ''},
                    user_id BIGINT,
                    knowledge_type TEXT, -- 'plugin', 'workflow', 'preference', 'fact', 'idea'
                    content TEXT,
                    context_snippet TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table for Reminders
            create_table('''
                CREATE TABLE IF NOT EXISTS user_reminders (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    reminder_text TEXT,
                    delay INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table for Notes
            create_table('''
                CREATE TABLE IF NOT EXISTS user_notes (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT,
                    note_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Table for Deleted Messages (The Snitch Engine)
            create_table('''
                CREATE TABLE IF NOT EXISTS deleted_messages (
                    id SERIAL PRIMARY KEY,
                    channel_id BIGINT,
                    user_id BIGINT,
                    username TEXT,
                    content TEXT,
                    attachments TEXT, -- JSON string
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Table for Guild Settings (Updates channel, prefixes, etc.)
            create_table('''
                CREATE TABLE IF NOT EXISTS guild_settings (
                    guild_id BIGINT PRIMARY KEY,
                    settings TEXT -- JSON string
                )
            ''')
            
            if not self.is_postgres:
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA synchronous=NORMAL')
            
            conn.commit()
            logger.info("Database initialized successfully.")
        finally:
            conn.close()

    # --- Conversation History ---
    def save_message(self, user_id, role, content):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'INSERT INTO conversation_history (user_id, role, content) VALUES ({p}, {p}, {p})',
                        (user_id, role, content)
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving message to DB: {e}")

    def get_history(self, user_id, limit=20):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'SELECT role, content FROM conversation_history WHERE user_id = {p} ORDER BY timestamp DESC LIMIT {p}',
                        (user_id, limit)
                    )
                    rows = cursor.fetchall()
                    return [{"role": row[0], "parts": [{"text": row[1]}]} for row in reversed(rows)]
        except Exception as e:
            logger.error(f"Error getting history from DB: {e}")
            return []

    # --- User Memory ---
    def get_user_memory(self, user_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'SELECT profile_summary, vibe, interaction_count FROM user_memory WHERE user_id = {p}',
                        (user_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        return {"profile_summary": row[0], "vibe": row[1], "interaction_count": row[2]}
                    return None
        except Exception as e:
            logger.error(f"Error getting user memory from DB: {e}")
            return None

    def update_user_memory(self, user_id, username, profile_summary=None, vibe=None, notes=None):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'SELECT interaction_count FROM user_memory WHERE user_id = {p}', (user_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        interaction_count = row[0] + 1
                        updates = ["interaction_count = %s", "last_updated = CURRENT_TIMESTAMP", "username = %s"]
                        if not self.is_postgres: updates = [u.replace('%s', '?') for u in updates]
                        
                        params = [interaction_count, username]
                        if profile_summary is not None: 
                            updates.append(f"profile_summary = {p}")
                            params.append(profile_summary)
                        if vibe is not None: 
                            updates.append(f"vibe = {p}")
                            params.append(vibe)
                        if notes is not None: 
                            updates.append(f"notes = {p}")
                            params.append(notes)
                        params.append(user_id)
                        
                        cursor.execute(f"UPDATE user_memory SET {', '.join(updates)} WHERE user_id = {p}", params)
                    else:
                        cursor.execute(
                            f'''INSERT INTO user_memory (user_id, username, profile_summary, vibe, notes, interaction_count) 
                               VALUES ({p}, {p}, {p}, {p}, {p}, 1)''',
                            (user_id, username, profile_summary or "New user", vibe or "neutral", notes or "")
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating user memory: {e}")

    # --- Levels ---
    def get_levels(self, guild_id=None):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if guild_id:
                        cursor.execute(f'SELECT user_id, xp, level FROM user_levels WHERE guild_id = {p}', (int(guild_id),))
                        return {row[0]: {"xp": row[1], "level": row[2]} for row in cursor.fetchall()}
                    else:
                        cursor.execute('SELECT guild_id, user_id, xp, level FROM user_levels')
                        levels = {}
                        for row in cursor.fetchall():
                            gid, uid, xp, lvl = row
                            if gid not in levels: levels[gid] = {}
                            levels[gid][uid] = {"xp": xp, "level": lvl}
                        return levels
        except Exception as e:
            logger.error(f"Error getting levels: {e}"); return {}

    def save_level(self, guild_id, user_id, xp, level):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if self.is_postgres:
                        cursor.execute(
                            'INSERT INTO user_levels (guild_id, user_id, xp, level) VALUES (%s, %s, %s, %s) ON CONFLICT (guild_id, user_id) DO UPDATE SET xp = EXCLUDED.xp, level = EXCLUDED.level',
                            (int(guild_id), user_id, xp, level)
                        )
                    else:
                        cursor.execute(
                            'INSERT OR REPLACE INTO user_levels (guild_id, user_id, xp, level) VALUES (?, ?, ?, ?)',
                            (int(guild_id), user_id, xp, level)
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving level: {e}")

    def get_user_level(self, guild_id, user_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'SELECT level FROM user_levels WHERE guild_id = {p} AND user_id = {p}', (int(guild_id), user_id))
                    row = cursor.fetchone()
                    return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error getting user level: {e}")
            return 0

    def get_leaderboard(self, guild_id, limit=10):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'SELECT user_id, xp, level FROM user_levels WHERE guild_id = {p} ORDER BY xp DESC LIMIT {limit}', (int(guild_id),))
                    return [{"user_id": row[0], "xp": row[1], "level": row[2]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []

    # --- Warnings ---
    def get_warnings(self):
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute('SELECT user_id, count, history FROM user_warnings')
                    return {str(row[0]): {"count": row[1], "history": json.loads(row[2])} for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting warnings: {e}"); return {}

    def save_warning(self, user_id, count, history):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if self.is_postgres:
                        cursor.execute(
                            'INSERT INTO user_warnings (user_id, count, history) VALUES (%s, %s, %s) ON CONFLICT (user_id) DO UPDATE SET count = EXCLUDED.count, history = EXCLUDED.history',
                            (int(user_id), count, json.dumps(history))
                        )
                    else:
                        cursor.execute(
                            'INSERT OR REPLACE INTO user_warnings (user_id, count, history) VALUES (?, ?, ?)',
                            (int(user_id), count, json.dumps(history))
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving warning: {e}")

    # --- YT Cooldowns ---
    def get_yt_cooldowns(self):
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute('SELECT user_id, expiry FROM yt_cooldowns')
                    return {str(row[0]): row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting yt cooldowns: {e}"); return {}

    def save_yt_cooldown(self, user_id, expiry):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if self.is_postgres:
                        cursor.execute(
                            'INSERT INTO yt_cooldowns (user_id, expiry) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET expiry = EXCLUDED.expiry',
                            (int(user_id), expiry)
                        )
                    else:
                        cursor.execute(
                            'INSERT OR REPLACE INTO yt_cooldowns (user_id, expiry) VALUES (?, ?)',
                            (int(user_id), expiry)
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving yt cooldown: {e}")

    # --- Guild Inviters ---
    def get_guild_inviters(self):
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute('SELECT guild_id, user_id FROM guild_inviters')
                    return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting guild inviters: {e}"); return {}

    def save_guild_inviter(self, guild_id, user_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if self.is_postgres:
                        cursor.execute(
                            'INSERT INTO guild_inviters (guild_id, user_id) VALUES (%s, %s) ON CONFLICT (guild_id) DO UPDATE SET user_id = EXCLUDED.user_id',
                            (str(guild_id), user_id)
                        )
                    else:
                        cursor.execute(
                            'INSERT OR REPLACE INTO guild_inviters (guild_id, user_id) VALUES (?, ?)',
                            (str(guild_id), user_id)
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving guild inviter: {e}")

    def delete_guild_inviter(self, guild_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'DELETE FROM guild_inviters WHERE guild_id = {p}', (str(guild_id),))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting guild inviter: {e}")

    # --- Portfolios ---
    def get_portfolios(self):
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute('SELECT user_id, portfolio_data FROM user_portfolios')
                    return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting portfolios: {e}"); return {}

    def save_portfolio(self, user_id, portfolio_data):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if self.is_postgres:
                        cursor.execute(
                            'INSERT INTO user_portfolios (user_id, portfolio_data) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET portfolio_data = EXCLUDED.portfolio_data',
                            (int(user_id), json.dumps(portfolio_data))
                        )
                    else:
                        cursor.execute(
                            'INSERT OR REPLACE INTO user_portfolios (user_id, portfolio_data) VALUES (?, ?)',
                            (int(user_id), json.dumps(portfolio_data))
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    # --- Captchas ---
    def get_active_captchas(self):
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute('SELECT user_id, code FROM active_captchas')
                    return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting captchas: {e}"); return {}

    def save_captcha(self, user_id, code):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    if self.is_postgres:
                        cursor.execute(
                            'INSERT INTO active_captchas (user_id, code) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET code = EXCLUDED.code, timestamp = CURRENT_TIMESTAMP',
                            (int(user_id), code)
                        )
                    else:
                        cursor.execute(
                            'INSERT OR REPLACE INTO active_captchas (user_id, code) VALUES (?, ?)',
                            (int(user_id), code)
                        )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving captcha: {e}")

    def delete_captcha(self, user_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'DELETE FROM active_captchas WHERE user_id = {p}', (int(user_id),))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting captcha: {e}")

    # --- Reminders ---
    def get_all_reminders(self):
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute('SELECT user_id, reminder_text, delay, timestamp FROM user_reminders')
                    return [{"user_id": row[0], "text": row[1], "delay": row[2], "timestamp": row[3]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting reminders: {e}"); return []

    def save_reminder(self, user_id, text, delay):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'INSERT INTO user_reminders (user_id, reminder_text, delay) VALUES ({p}, {p}, {p})',
                        (int(user_id), text, delay)
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving reminder: {e}")

    def delete_reminder(self, user_id, text):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'DELETE FROM user_reminders WHERE user_id = {p} AND reminder_text = {p}', (int(user_id), text))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting reminder: {e}")

    # --- Notes ---
    def get_notes(self, user_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'SELECT note_text FROM user_notes WHERE user_id = {p} ORDER BY timestamp DESC', (int(user_id),))
                    return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting notes: {e}"); return []

    def save_note(self, user_id, text):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'INSERT INTO user_notes (user_id, note_text) VALUES ({p}, {p})',
                        (int(user_id), text)
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving note: {e}")

    def delete_notes(self, user_id):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'DELETE FROM user_notes WHERE user_id = {p}', (int(user_id),))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting notes: {e}")

    # --- Deleted Messages (The Snitch Engine) ---
    def save_deleted_message(self, channel_id, user_id, username, content, attachments):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'''INSERT INTO deleted_messages (channel_id, user_id, username, content, attachments) 
                           VALUES ({p}, {p}, {p}, {p}, {p})''',
                        (channel_id, user_id, username, content, json.dumps(attachments))
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving deleted message: {e}")

    def get_latest_deleted_messages(self, channel_id, limit=3):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(
                        f'SELECT user_id, username, content, attachments, timestamp FROM deleted_messages WHERE channel_id = {p} ORDER BY timestamp DESC LIMIT {p}',
                        (channel_id, limit)
                    )
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting deleted messages: {e}")
            return []

    # --- Guild Settings ---
    def save_guild_setting(self, guild_id, key, value):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    # Get existing settings
                    cursor.execute(f'SELECT settings FROM guild_settings WHERE guild_id = {p}', (int(guild_id),))
                    row = cursor.fetchone()
                    settings = json.loads(row[0]) if row else {}
                    
                    # Update setting
                    settings[key] = value
                    
                    # Save back
                    if row:
                        cursor.execute(f'UPDATE guild_settings SET settings = {p} WHERE guild_id = {p}', (json.dumps(settings), int(guild_id)))
                    else:
                        cursor.execute(f'INSERT INTO guild_settings (guild_id, settings) VALUES ({p}, {p})', (int(guild_id), json.dumps(settings)))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving guild setting: {e}")

    def get_guild_setting(self, guild_id, key, default=None):
        p = self.get_placeholder()
        try:
            with self.get_connection() as conn:
                with self.get_cursor(conn) as cursor:
                    cursor.execute(f'SELECT settings FROM guild_settings WHERE guild_id = {p}', (int(guild_id),))
                    row = cursor.fetchone()
                    if row:
                        settings = json.loads(row[0])
                        return settings.get(key, default)
                    return default
        except Exception as e:
            logger.error(f"Error getting guild setting: {e}")
            return default

    def add_to_brain(self, user_id, knowledge_type, content, context_snippet=None):
        """Add a specific item to the user's second brain."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cursor:
                query = '''
                    INSERT INTO user_brain (user_id, knowledge_type, content, context_snippet)
                    VALUES (%s, %s, %s, %s)
                ''' if self.is_postgres else '''
                    INSERT INTO user_brain (user_id, knowledge_type, content, context_snippet)
                    VALUES (?, ?, ?, ?)
                '''
                cursor.execute(query, (user_id, knowledge_type, content, context_snippet))
                conn.commit()

    def get_brain(self, user_id, limit=20):
        """Retrieve the latest knowledge items from the user's brain."""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cursor:
                query = '''
                    SELECT knowledge_type, content, added_at 
                    FROM user_brain 
                    WHERE user_id = %s 
                    ORDER BY added_at DESC LIMIT %s
                ''' if self.is_postgres else '''
                    SELECT knowledge_type, content, added_at 
                    FROM user_brain 
                    WHERE user_id = ? 
                    ORDER BY added_at DESC LIMIT ?
                '''
                cursor.execute(query, (user_id, limit))
                results = cursor.fetchall()
                
                brain_items = []
                for row in results:
                    brain_items.append({
                        "type": row[0],
                        "content": row[1],
                        "added_at": row[2]
                    })
                return brain_items

db_manager = DatabaseManager()
