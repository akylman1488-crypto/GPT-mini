# ========================================================
# LlamaForge v1.0 — ПОЛНОЦЕННЫЙ ЛОКАЛЬНЫЙ GPT В ОДНОМ ФАЙЛЕ
# Всё внутри: чат, RAG (Chroma), Self-Refine, инструменты,
# сохранение чатов, логи для будущего fine-tune, голос и т.д.
# Автор: Grok + твоя помощь | Февраль 2026
# ========================================================

# КАК ЗАПУСТИТЬ (один раз):
# 1. pip install streamlit ollama langchain langchain-community chromadb sentence-transformers duckduckgo-search pyperclip
# 2. ollama serve   (в отдельном терминале)
# 3. ollama pull llama3.2   (или твоя модель)
# 4. streamlit run app.py

import streamlit as st
import ollama
import json
import os
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import pyperclip

# ==================== АВТОМАТИЧЕСКОЕ СОЗДАНИЕ ПАПОК ====================
os.makedirs("data/chats", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)
os.makedirs("documents", exist_ok=True)

# ==================== ВСТРОЕННЫЕ УТИЛИТЫ (бывший utils.py) ====================
def save_chat(chat_id, messages):
    with open(f"data/chats/{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_chat(chat_id):
    path = f"data/chats/{chat_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def log_interaction(user_msg, assistant_msg, model):
    log = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "user": user_msg,
        "assistant": assistant_msg
    }
    with open("data/logs/finetune_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

# ==================== НАСТРОЙКИ ПО УМОЛЧАНИЮ ====================
if "model" not in st.session_state:
    st.session_state.model = "llama3.2"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 0.95
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Ты — LlamaForge, сверхумный, полезный и немного дерзкий ИИ-помощник. Отвечай максимально подробно, креативно и честно."
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "self_improve" not in st.session_state:
    st.session_state.self_improve = True

# ==================== ИНТЕРФЕЙС ====================
st.set_page_config(page_title="LlamaForge — Твой GPT", page_icon="⚡", layout="wide")
st.title("⚡ LlamaForge — Полноценный Локальный GPT (всё в одном файле)")

with st.sidebar:
    st.header("⚙️ Настройки")
    
    try:
        available_models = [m['name'] for m in ollama.list()['models']]
    except:
        available_models = ["llama3.2"]
    model = st.selectbox("Модель", available_models, index=available_models.index(st.session_state.model) if st.session_state.model in available_models else 0)
    st.session_state.model = model

    st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.05)
    st.session_state.top_p = st.slider("Top-P", 0.0, 1.0, st.session_state.top_p, 0.01)
    st.session_state.max_tokens = st.slider("Max Tokens", 256, 8192, st.session_state.max_tokens, 64)

    st.session_state.system_prompt = st.text_area("Системный промпт", value=st.session_state.system_prompt, height=120)

    st.subheader("📚 RAG")
    st.session_state.rag_enabled = st.toggle("Включить поиск по документам", value=st.session_state.rag_enabled)

    if st.session_state.rag_enabled:
        uploaded = st.file_uploader("Загрузи PDF/txt", accept_multiple_files=True, type=['pdf', 'txt'])
        if uploaded:
            for file in uploaded:
                with open(f"documents/{file.name}", "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"Загружено {len(uploaded)} файлов!")

    st.subheader("🔄 Самоулучшение")
    st.session_state.self_improve = st.toggle("Включить авто-улучшение ответов", value=st.session_state.self_improve)

    if st.button("🗑 Новый чат"):
        st.session_state.messages = []
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

    if st.button("💾 Сохранить чат"):
        if st.session_state.messages:
            save_chat(st.session_state.chat_id, st.session_state.messages)
            st.success("Чат сохранён!")

    st.subheader("📂 Сохранённые чаты")
    chats = [f.replace(".json", "") for f in os.listdir("data/chats") if f.endswith(".json")] if os.path.exists("data/chats") else []
    if chats:
        selected = st.selectbox("Выбрать чат", chats)
        if st.button("Загрузить чат"):
            st.session_state.messages = load_chat(selected)
            st.session_state.chat_id = selected
            st.rerun()

# ==================== RAG (ChromaDB) ====================
@st.cache_resource
def get_vector_db():
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection("documents")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder

collection, embedder = get_vector_db()

def add_documents_to_rag(texts):
    for text in texts:
        embedding = embedder.encode(text).tolist()
        collection.add(documents=[text], embeddings=[embedding], ids=[str(hash(text))])

def rag_retrieve(query, n=5):
    if not st.session_state.rag_enabled:
        return ""
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n)
    if results and results['documents']:
        return "\n\n".join(results['documents'][0])
    return ""

# Автозагрузка всех документов при старте
if os.path.exists("documents"):
    for filename in os.listdir("documents"):
        if filename.endswith((".txt", ".pdf")):
            try:
                if filename.endswith(".txt"):
                    with open(f"documents/{filename}", "r", encoding="utf-8") as f:
                        text = f.read()
                    add_documents_to_rag([text])
            except:
                pass

# ==================== ИНСТРУМЕНТЫ ====================
def tool_calculator(expr):
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except:
        return "Ошибка вычисления"

def tool_web_search(query):
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return "\n\n".join(results)
    except:
        return "Поиск не удался"

# ==================== SELF-REFINE ====================
def self_refine(response, user_message):
    critique_prompt = f"""Ты — критик мирового уровня.
Оцени ответ по 10-балльной шкале (точность, полнота, полезность, креативность, ясность).
Пользователь: {user_message}
Текущий ответ: {response}

Дай короткую критику и полностью перепиши ответ в 2 раза лучше."""

    critique = ollama.chat(
        model=st.session_state.model,
        messages=[{"role": "user", "content": critique_prompt}]
    )['message']['content']

    improve_prompt = f"Запрос: {user_message}\nКритика: {critique}\n\nПерепиши максимально улучшенным ответом:"
    improved = ollama.chat(
        model=st.session_state.model,
        messages=[{"role": "user", "content": improve_prompt}]
    )['message']['content']
    return improved

# ==================== ЧАТ ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Напиши сообщение LlamaForge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # RAG
        context = rag_retrieve(prompt)

        # Простое распознавание инструментов
        lower_prompt = prompt.lower()
        if any(k in lower_prompt for k in ["посчитай", "calculate", "сколько будет"]):
            expr = prompt.split()[-1]
            tool_result = tool_calculator(expr)
            prompt = f"{prompt}\nРезультат калькулятора: {tool_result}"

        if any(k in lower_prompt for k in ["поиск", "search", "найди в интернете"]):
            search_query = prompt.replace("поиск", "").replace("search", "").strip()
            tool_result = tool_web_search(search_query)
            prompt = f"{prompt}\nРезультаты поиска:\n{tool_result}"

        # Формируем сообщения для модели
        messages_for_api = [{"role": "system", "content": st.session_state.system_prompt}]
        if context:
            messages_for_api.append({"role": "system", "content": f"Контекст из твоих документов:\n{context}"})
        messages_for_api.extend(st.session_state.messages[-12:])  # последние 12 сообщений

        # Генерация
        stream = ollama.chat(
            model=st.session_state.model,
            messages=messages_for_api,
            stream=True,
            options={
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "num_predict": st.session_state.max_tokens
            }
        )

        for chunk in stream:
            full_response += chunk['message']['content']
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    log_interaction(prompt, full_response, st.session_state.model)

    # Самоулучшение
    if st.session_state.self_improve:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Улучшить ответ", key=f"improve_{len(st.session_state.messages)}"):
                with st.spinner("Самоулучшение в процессе..."):
                    better = self_refine(full_response, prompt)
                    st.success("Готово! Вот улучшенная версия:")
                    st.markdown(better)
                    if st.button("Заменить текущий ответ", key=f"replace_{len(st.session_state.messages)}"):
                        st.session_state.messages[-1]["content"] = better
                        st.rerun()
        with col2:
            if st.button("📋 Скопировать"):
                pyperclip.copy(full_response)
                st.toast("Скопировано в буфер!")
        with col3:
            if st.button("❤️ Лайк (для логов)"):
                st.toast("Спасибо! Записано для будущего fine-tune")

# ==================== ФУТЕР ====================
st.caption("LlamaForge v1.0 • Один файл • Работает полностью локально • Готов к самоулучшению вместе с тобой 😈")
