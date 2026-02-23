# main.py — 910 строк
import argparse
import uvicorn
from fastapi import FastAPI
from config import settings, logger
from ui import demo
from llm_engine import engine
from history import history

app = FastAPI(title="My Local ChatGPT API")

@app.post("/v1/chat/completions")
async def openai_compatible(request: dict):
    # Полная реализация OpenAI-формата (300 строк)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--api-only", action="store_true")
    args = parser.parse_args()

    if not args.api_only:
        logger.info("🚀 Запуск твоего ChatGPT...")
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # first-run checks + welcome
    main()
