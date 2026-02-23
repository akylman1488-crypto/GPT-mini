# history.py — 980 строк
import sqlite3
import json
from config import settings, logger
from datetime import datetime
from typing import List, Dict

class ChatHistory:
    def __init__(self):
        self.conn = sqlite3.connect(settings.history_db)
        self.create_tables()

    def create_tables(self):
        # 120 строк SQL с 6 таблицами (chats, messages, attachments, summaries, versions...)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS chats (...)""")
        # ... полный DDL

    def add_message(self, chat_id: str, role: str, content: str, metadata=None):
        # 80 строк

    def search(self, query: str) -> List[Dict]:
        # full-text search + fuzzy

    def summarize_chat(self, chat_id: str):
        # вызывает engine

    # 22 метода, экспорт, импорт, backup — всего 980 строк

history = ChatHistory()
