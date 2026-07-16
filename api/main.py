# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import time

import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from adapters import OpenAICompatAdapter
from crisis import detect_crisis, crisis_response
from dual_engine import DualEngine

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.5")
GROK_KEY = os.getenv("GROK_API_KEY", "")

DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MOCK_MODE = os.getenv("AMIHERBAI_MOCK", "false").lower() == "true"

STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SITE_URL = os.getenv("SITE_URL", "https://amiherbai.com")

stripe.api_key = STRIPE_SECRET

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "https://amiherbai.com,http://localhost:8000"
).split(",")

SYSTEM_PROMPT = (
    "You are amiHerbAI — an honest, warm, knowledgeable companion for herbal medicine "
    "and phytotherapy. You are an AI system (not a human, doctor, or pharmacist) and you "
    "say so if asked. You speak in the user's language (detect from their message).\n\n"
    "CRITICAL RULES:\n"
    "- You are NOT a doctor, pharmacist, or healthcare provider.\n"
    "- You do NOT diagnose, prescribe, or recommend treatments.\n"
    "- You provide INFORMATIONAL content about traditional and evidence-based herbal knowledge.\n"
    "- For any health concern, always suggest consulting a qualified healthcare professional.\n"
    "- When discussing dosages, clearly state these are traditional/literature references, not prescriptions.\n"
    "- Always mention known contraindications and herb-drug interactions when relevant.\n"
    "- If unsure, say 'I don't know' rather than guessing.\n"
    "- Keep responses concise (3-5 sentences) unless the user asks for detail.\n"
    "- Include a brief safety note when discussing any herb that has known risks."
)

RATE_LIMIT: dict[str, list[float]] = {}


def _rate_ok(ip: str, max_req: int = 20, window: float = 60.0) -> bool:
    now = time.time()
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT.get(ip, []) if now - t < window] + [now]
    return len(RATE_LIMIT[ip]) <= max_req


def _sign(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build_engine() -> DualEngine | None:
    if MOCK_MODE or not GROK_KEY or not DEEPSEEK_KEY:
        return None
    grok = OpenAICompatAdapter(
        base_url=GROK_URL, model=GROK_MODEL, api_key=GROK_KEY,
        temperature=0.3, max_tokens=400, timeout=30, name="grok",
    )
    deepseek = OpenAICompatAdapter(
        base_url=DEEPSEEK_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_KEY,
        temperature=0.3, max_tokens=400, timeout=30, name="deepseek",
    )
    return DualEngine(grok, deepseek, system_prompt=SYSTEM_PROMPT, threshold=0.45)


ENGINE = _build_engine()

app = FastAPI(title="amiHerbAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    locale: str = "EN"


class ChatResponse(BaseModel):
    response: str
    engine: str
    decision: str
    certified: bool
    signature: str
    is_crisis_response: bool
    concordance: float | None = None
    latency_s: float = 0.0


@app.get("/")
def root():
    return {"service": "amiHerbAI API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "dual (Grok + DeepSeek)" if ENGINE else "mock",
        "grok_configured": bool(GROK_KEY),
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "stripe_configured": bool(STRIPE_SECRET and STRIPE_PRICE_ID),
        "mock_mode": MOCK_MODE or not ENGINE,
    }


@app.post("/companion/herb/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many requests — please wait a moment")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 1000:
        raise HTTPException(400, "Message too long (max 1000 chars)")

    locale = req.locale.upper()[:2] if req.locale else "EN"

    crisis = detect_crisis(msg)
    if crisis.is_crisis:
        resp = crisis_response(locale.lower())
        return ChatResponse(
            response=resp, engine="safety-layer", decision="crisis-intercept",
            certified=True, signature=_sign(resp), is_crisis_response=True,
        )

    if ENGINE:
        result = ENGINE.ask(msg)
        return ChatResponse(
            response=result.reply, engine=result.engine, decision=result.decision,
            certified=True, signature=_sign(result.reply), is_crisis_response=False,
            concordance=result.concordance, latency_s=result.latency_s,
        )

    mock = "That's an interesting question about herbs! In a production setup, I'd cross-verify this with two AI engines. Please check back soon."
    return ChatResponse(
        response=mock, engine="mock", decision="mock",
        certified=False, signature=_sign(mock), is_crisis_response=False,
    )


# --- Stripe endpoints ---

@app.post("/create-checkout-session")
def create_checkout():
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise HTTPException(503, "Payment system not configured yet")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=SITE_URL + "?payment=success",
            cancel_url=SITE_URL + "?payment=cancelled",
        )
    except stripe.error.StripeError as e:
        raise HTTPException(502, f"Payment provider error: {e.user_message or str(e)}")
    return {"url": session.url}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        print(f"[STRIPE] New subscription: {session.get('customer_email', 'unknown')}")
    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        print(f"[STRIPE] Subscription cancelled: {sub.get('id')}")

    return {"received": True}


# --- GDPR endpoints ---

@app.post("/gdpr/data-request")
def gdpr_data_request():
    return {
        "message": "amiHerbAI does not store chat messages on its servers. "
                   "Chat data is processed in real-time and discarded. "
                   "For subscription data managed by Stripe, email privacy@amiherbai.com.",
        "data_stored": "none (stateless chat processing)",
    }


@app.delete("/gdpr/delete")
def gdpr_delete():
    return {
        "message": "No personal data to delete — amiHerbAI processes chats statelessly. "
                   "For Stripe subscription deletion, email privacy@amiherbai.com.",
        "status": "no_data_held",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
