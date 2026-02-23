# ========================================================
# LlamaForge v2.0 — ПОЛНОЦЕННЫЙ GPT ДЛЯ STREAMLIT CLOUD НА GROQ
# Всё в одном файле: RAG, самоулучшение, инструменты, чаты, логи
# Автор: Grok + ты | Февраль 2026
# ========================================================

import streamlit as st
from groq import Groq
import json
import os
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import pyperclip

st.set_page_config(page_title="LlamaForge Cloud", page_icon="⚡", layout="wide")
st.title("⚡ LlamaForge Cloud — Полноценный GPT на Groq")

# ==================== API КЛЮЧ ====================
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.error("Добавь ключ в Secrets или введи здесь")
    st.stop()

client = Groq(api_key=api_key)

# ==================== ПАПКИ ====================
os.makedirs("data/chats", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)
os.makedirs("documents", exist_ok=True)

# ==================== УТИЛИТЫ ====================
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
    log = {"timestamp": datetime.now().isoformat(), "model": model, "user": user_msg, "assistant": assistant_msg}
    with open("data/logs/finetune_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

# ==================== НАСТРОЙКИ ====================
if "model" not in st.session_state:
    st.session_state.model = "llama-3.3-70b-versatile"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Ты — LlamaForge Cloud, сверхумный, полезный и дерзкий ИИ. Отвечай подробно и креативно."
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "self_improve" not in st.session_state:
    st.session_state.self_improve = True

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Настройки LlamaForge Cloud")
    models = {
        "Llama-3.3-70B (самая мощная)": "llama-3.3-70b-versatile",
        "Llama-3.1-70B": "llama-3.1-70b-versatile",
        "Llama-3.1-8B (быстрая)": "llama-3.1-8b-instant"
    }
    selected = st.selectbox("Модель", list(models.keys()), index=0)
    st.session_state.model = models[selected]

    st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.05)
    st.session_state.max_tokens = st.slider("Max Tokens", 256, 8192, st.session_state.max_tokens, 64)

    st.session_state.system_prompt = st.text_area("Системный промпт", value=st.session_state.system_prompt, height=100)

    st.subheader("📚 RAG")
    st.session_state.rag_enabled = st.toggle("Включить RAG", value=st.session_state.rag_enabled)

    uploaded = st.file_uploader("Загрузи txt/pdf", accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            with open(f"documents/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Загружено {len(uploaded)} файлов!")

    st.subheader("🔄 Самоулучшение")
    st.session_state.self_improve = st.toggle("Авто-улучшение", value=st.session_state.self_improve)

    if st.button("🗑 Новый чат"):
        st.session_state.messages = []
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

    if st.button("💾 Сохранить чат"):
        if st.session_state.messages:
            save_chat(st.session_state.chat_id, st.session_state.messages)
            st.success("Сохранено!")

    chats = [f.replace(".json","") for f in os.listdir("data/chats") if f.endswith(".json")] if os.path.exists("data/chats") else []
    if chats:
        selected_chat = st.selectbox("Загрузить чат", chats)
        if st.button("Загрузить"):
            st.session_state.messages = load_chat(selected_chat)
            st.session_state.chat_id = selected_chat
            st.rerun()

# ==================== RAG ====================
@st.cache_resource
def get_vector_db():
    client_db = chromadb.PersistentClient(path="data/chroma_db")
    collection = client_db.get_or_create_collection("docs")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder

collection, embedder = get_vector_db()

def rag_retrieve(query, n=5):
    if not st.session_state.rag_enabled:
        return ""
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n)
    if results['documents']:
        return "\n\n".join(results['documents'][0])
    return ""

# Автозагрузка документов
if os.path.exists("documents"):
    for filename in os.listdir("documents"):
        if filename.endswith(".txt"):
            with open(f"documents/{filename}", "r", encoding="utf-8") as f:
                text = f.read()
            embedding = embedder.encode(text).tolist()
            collection.add(documents=[text], embeddings=[embedding], ids=[filename])

# ==================== ИНСТРУМЕНТЫ ====================
def tool_web_search(query):
    try:
        with DDGS() as ddgs:
            return "\n\n".join([r['body'] for r in ddgs.text(query, max_results=3)])
    except:
        return "Поиск недоступен"

# ==================== SELF-REFINE ====================
def self_refine(response, user_message):
    critique_prompt = f"Оцени ответ по 10-балльной шкале и перепиши его в 2 раза лучше. Запрос: {user_message}\nОтвет: {response}"
    critique = client.chat.completions.create(
        model=st.session_state.model,
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.7,
        max_tokens=1024
    ).choices[0].message.content
    return critique

# ==================== ЧАТ ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Напиши сообщение LlamaForge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        context = rag_retrieve(prompt) if st.session_state.rag_enabled else ""

        messages_api = [{"role": "system", "content": st.session_state.system_prompt}]
        if context:
            messages_api.append({"role": "system", "content": f"Контекст:\n{context}"})
        messages_api += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]

        # Инструменты
        if any(k in prompt.lower() for k in ["поиск", "найди", "search"]):
            search_q = prompt.replace("поиск", "").strip()
            tool_res = tool_web_search(search_q)
            messages_api.append({"role": "system", "content": f"Результаты поиска:\n{tool_res}"})

        stream = client.chat.completions.create(
            model=st.session_state.model,
            messages=messages_api,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    log_interaction(prompt, full_response, st.session_state.model)

    # Самоулучшение
    if st.session_state.self_improve:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Улучшить ответ", key=f"imp_{len(st.session_state.messages)}"):
                with st.spinner("Самоулучшение..."):
                    better = self_refine(full_response, prompt)
                    st.success("Улучшено!")
                    st.markdown(better)
                    if st.button("Заменить", key=f"rep_{len(st.session_state.messages)}"):
                        st.session_state.messages[-1]["content"] = better
                        st.rerun()
        with col2:
            if st.button("📋 Скопировать"):
                pyperclip.copy(full_response)
                st.toast("Скопировано!")

st.caption("LlamaForge Cloud v2.0 • Groq + RAG + Self-Improve • Работает онлайн на твоём сайте • 2026")
