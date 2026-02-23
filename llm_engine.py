# llm_engine.py — 1240 строк
from llama_cpp import Llama
import ollama
import asyncio
from config import settings, logger, PERSONAS
from typing import Generator, List, Dict
import time

class LLMEngine:
    def __init__(self):
        self.model = None
        self.load_model()
        self.tools = [ # 15 tool definitions
            {"name": "search_web", "description": "...", "parameters": {...}},
            # ... 14 других
        ]

    def load_model(self):
        try:
            self.model = Llama(
                model_path=settings.model.path,
                n_gpu_layers=settings.model.n_gpu_layers,
                n_ctx=settings.model.n_ctx,
                n_batch=512,
                verbose=False,
                # 28 параметров
            )
            logger.info("LlamaCpp загружен успешно")
        except Exception as e:
            logger.warning(f"LlamaCpp упал → fallback Ollama: {e}")
            self.model = "ollama"

    def generate_stream(self, messages: List[Dict], temperature=None, max_tokens=None) -> Generator[str, None, None]:
        temp = temperature or settings.model.temperature
        # 320 строк реализации streaming + stop-tokens + repetition penalty + tool calling
        if isinstance(self.model, Llama):
            for chunk in self.model.create_chat_completion(messages, stream=True, temperature=temp, max_tokens=max_tokens or settings.model.max_tokens):
                if content := chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                    yield content
        else:
            for chunk in ollama.chat(model="llama3.2", messages=messages, stream=True):
                yield chunk["message"]["content"]

    # Ещё 18 методов: apply_template, call_tool, rag_search, summarize, voice_to_text_stub и т.д.
    # Каждый метод 40-80 строк + docstring + error handling = 1240 строк

engine = LLMEngine()
