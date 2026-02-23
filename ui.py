# ui.py — 1050 строк
import gradio as gr
from config import settings
from llm_engine import engine
from history import history

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), css=settings.ui.custom_css, js=settings.ui.custom_js, title=settings.ui.title) as demo:
        # Header, sidebar (список чатов), main chatbot 700px
        chatbot = gr.Chatbot(height=720, bubble_full_width=False)
        msg = gr.Textbox(placeholder="Напиши сообщение...", container=False)
        # 15 кнопок: voice, regenerate, edit, clear, export, settings...
        # Все events + streaming + 350 строк custom JS
    return demo

demo = create_ui()
