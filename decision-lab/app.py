"""Loopback-only, offline GLiNER decision playground. No VDMA imports or actions."""
import copy
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from download_model import MODEL_DIR, MODEL_ID, REVISION, ROOT

os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    state: str = Field(min_length=1, max_length=6000)
    question: str = Field(min_length=1, max_length=300)
    options: list[str] = Field(min_length=2, max_length=12)

    @field_validator("options")
    @classmethod
    def valid_options(cls, values):
        values = [value.strip() for value in values]
        if any(not value or len(value) > 80 for value in values):
            raise ValueError("Use 2–12 nonempty options, each at most 80 characters.")
        if len({value.casefold() for value in values}) != len(values):
            raise ValueError("Options must be distinct.")
        return values


def decision_schema(model, payload):
    # Return every label using the same softmax as native single-choice mode.
    # multi_label only controls output selection here; it does NOT use sigmoid.
    return model.create_schema().classification(
        "answer", payload.options, prompt=payload.question,
        multi_label=True, class_act="softmax", cls_threshold=0.0,
    )


class DecisionEngine:
    def __init__(self):
        self.model = None
        self.status = "loading"
        self.error = None
        # ponytail: one bounded inference slot; use workers only if throughput warrants it.
        self.lock = threading.Lock()

    def load(self):
        try:
            if not (MODEL_DIR / "model.safetensors").is_file():
                raise RuntimeError("Model files missing. Run ./setup.sh, then restart.")
            if (MODEL_DIR / "revision.txt").read_text().strip() != REVISION:
                raise RuntimeError("Model revision differs. Run ./setup.sh, then restart.")
            import torch
            from gliner2 import AutoExtractor

            torch.set_num_threads(2)
            self.model = AutoExtractor.from_pretrained(str(MODEL_DIR), local_files_only=True, map_location="cpu")
            self.model.eval()
            self.model.processor.change_mode(is_training=False)
            self.status = "ready"
        except Exception:
            self.model = None
            self.status = "error"
            self.error = "Could not load the local model. Check the server log and run ./setup.sh."
            logging.exception("Local model load failed")

    def decide(self, payload):
        if self.status != "ready":
            raise HTTPException(503, self.error or "The model is loading. Try again shortly.")
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429, "A decision is running. Try again when it finishes.")
        try:
            model = self.model
            # Reject reserved tokenizer markers rather than corrupting label alignment.
            values = [payload.state, payload.question, *payload.options]
            if any(marker in value for marker in model.processor.tokenizer.all_special_tokens for value in values):
                raise HTTPException(422, "Inputs cannot contain reserved model tokens.")
            schema = decision_schema(model, payload)
            record = model.processor._transform_record({"text": payload.state, "schema": copy.deepcopy(schema.build())})
            tokens = len(record.input_ids)
            if tokens > 512:
                raise HTTPException(422, f"This request uses {tokens} tokens including its question and options. Shorten it to 512 tokens or fewer; nothing was truncated.")
            started = time.perf_counter()
            result = model.extract(payload.state, schema, format_results=False, include_confidence=True)
            elapsed = (time.perf_counter() - started) * 1000
            pairs = result["answer"]
            if len(pairs) != len(payload.options) or {p[0] for p in pairs} != set(payload.options):
                raise RuntimeError("Model label alignment failed")
            scores = [{"label": label, "score": float(score)} for label, score in pairs]
            if any(not math.isfinite(x["score"]) or not 0 <= x["score"] <= 1 for x in scores):
                raise RuntimeError("Invalid model scores")
            if not math.isclose(sum(x["score"] for x in scores), 1.0, abs_tol=1e-5):
                raise RuntimeError("Invalid softmax distribution")
            scores.sort(key=lambda x: -x["score"])
            return {
                "answer": scores[0]["label"], "scores": scores,
                "latency_ms": round(elapsed, 1), "input_tokens": tokens,
                "margin": scores[0]["score"] - scores[1]["score"],
                "model": MODEL_ID, "revision": REVISION, "device": "cpu",
                "score_type": "softmax", "calibrated": False,
            }
        except HTTPException:
            raise
        except Exception:
            # Never log user text or model output.
            logging.error("Decision inference failed")
            raise HTTPException(500, "Local inference failed. Check the runtime and try a shorter request.") from None
        finally:
            self.lock.release()


engine = DecisionEngine()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=engine.load, name="decision-model-load", daemon=True).start()
    yield


app = FastAPI(title="Decision Lab", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        expected = f"http://{request.headers.get('host', '')}"
        if (origin and origin != expected) or request.headers.get("x-decision-client") != "local-ui":
            return JSONResponse({"detail": "Use the local Decision Lab UI or its documented API header."}, status_code=403)
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            return JSONResponse({"detail": "Send application/json."}, status_code=415)
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > 32768:
                return JSONResponse({"detail": "Request too large."}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    messages = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
    return JSONResponse({"detail": " ".join(messages)}, status_code=422)


@app.get("/api/status")
def status():
    return {"status": engine.status, "error": engine.error, "model": MODEL_ID, "revision": REVISION, "device": "cpu", "offline": True}


@app.post("/api/decide")
def decide(payload: DecisionRequest):
    return engine.decide(payload)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/explainer")
def explainer():
    return FileResponse(ROOT / "static" / "explainer.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8771, access_log=False)
