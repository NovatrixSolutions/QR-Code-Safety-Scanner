"""
app.py - FraudBlocker AI
Full FastAPI app with:
 - URL / message / QR text inspection
 - QR image decoding (pyzbar + OpenCV fallback)
 - Email (RFC822) inspection
 - TF-IDF + Logistic (classical NLP) baseline
 - Tiny Keras model (local deep baseline)
 - Heuristics for URLs and emails
 - Optional Gemini (google-genai) fusion (set GEMINI_API_KEY to use)
"""

# -------------------------
# Standard library imports
# -------------------------
import os
import re
import json
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from email import message_from_string, policy

# -------------------------
# Third-party imports
# -------------------------
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

# ML / NLP / DL libs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras import layers

# Image / QR libs (Pillow + pyzbar/opencv)
from PIL import Image

# Gemini / Google GenAI SDK (used only if GEMINI_API_KEY set)
# We import lazily in functions to keep start-up robust if SDK missing.
try:
    from google import genai
    from google.genai import types as gtypes
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# -------------------------
# >>> CONFIG / CONSTANTS
# -------------------------
GEMINI_API_KEY = "AIzaSyDqX1f-rpLOBRLbWKlw-s3VQ7sPEenlKCw"

GEMINI_MODEL = "gemini-2.5-flash"

# Suspicious TLD set for simple heuristics
SUSPICIOUS_TLDS = {".ru", ".tk", ".cn", ".ml", ".ga", ".gq"}

# Keywords used for engineered features / heuristics
PHISH_KEYWORDS = [
    "kyc","update","verify","urgent","immediately","otp","limited","refund","reward",
    "investment","double","gift","lottery","block","suspend","approve","verification",
]

# -------------------------
# >>> Utility helpers
# -------------------------
def sha256(s: str) -> str:
    """Return hex sha256 of input string (used in artifacts)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -------------------------
# >>> NLP: TF-IDF + Logistic baseline
# -------------------------
# Seed examples to bootstrap the vectorizer and logistic model (quick demo)
NLP_SEED_SUSPICIOUS = [
    "urgent your account will be blocked verify now",
    "win lottery claim reward click link",
    "update kyc immediately to avoid suspension",
    "refund pending complete payment to receive",
    "upi request from unknown sender approve now",
    "security alert unusual login provide otp",
    "investment double money guaranteed",
    "official notice pay service fee immediately",
    "free gift limited time click here",
    "scam attempt payment link phishing"
]
NLP_SEED_SAFE = [
    "payment received successfully thank you",
    "your order has been dispatched",
    "upi payment to trusted merchant completed",
    "monthly newsletter updates and offers",
    "meeting at 3pm see you",
    "your phone bill receipt is attached",
]

# Vectorizer and classifier (fitted on small seed set)
VEC = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=6000)
VEC.fit(NLP_SEED_SUSPICIOUS + NLP_SEED_SAFE)
NLP_CLS = LogisticRegression(max_iter=1000)
X_small = VEC.transform(NLP_SEED_SUSPICIOUS + NLP_SEED_SAFE)
y_small = [1]*len(NLP_SEED_SUSPICIOUS) + [0]*len(NLP_SEED_SAFE)
NLP_CLS.fit(X_small, y_small)

def nlp_score(text: str) -> float:
    """Return suspicious probability from TF-IDF + Logistic baseline."""
    try:
        vec = VEC.transform([text.lower()])
        return float(NLP_CLS.predict_proba(vec)[0][1])
    except Exception:
        return 0.5

# -------------------------
# >>> Deep (tiny Keras) baseline
# -------------------------
def dl_features(text: str) -> np.ndarray:
    """Extract simple engineered features for the tiny deep model."""
    t = text.lower()
    feats = [
        len(t) / 500.0,                                  # normalized length
        sum(k in t for k in PHISH_KEYWORDS) / 10.0,      # keyword density
        t.count("http") / 3.0,                           # link count heuristic
        int(any(c.isdigit() for c in t)) * 0.3,          # digits present
        int("upi" in t) * 0.4,                           # mentions UPI
    ]
    return np.array(feats, dtype="float32")[None, ...]

# Build tiny deep model (very small; demo only)
DL_INPUT_DIM = 5
DEEP_MODEL = keras.Sequential([
    layers.Input(shape=(DL_INPUT_DIM,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
DEEP_MODEL.compile(optimizer="adam", loss="binary_crossentropy")

# Quick synthetic fit so outputs are non-trivial
X_train_dl = np.vstack([dl_features(t)[0] for t in (NLP_SEED_SUSPICIOUS + NLP_SEED_SAFE)])
y_train_dl = np.array([1 if t in NLP_SEED_SUSPICIOUS else 0 for t in (NLP_SEED_SUSPICIOUS + NLP_SEED_SAFE)], dtype="float32")
DEEP_MODEL.fit(X_train_dl, y_train_dl, epochs=10, verbose=0)

def deep_score(text: str) -> float:
    """Return deep model probability (0-1)."""
    try:
        return float(DEEP_MODEL.predict(dl_features(text), verbose=0)[0][0])
    except Exception:
        return 0.5

# -------------------------
# >>> URL heuristics & helpers
# -------------------------
def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Parse URL and return simple boolean/numeric features used by heuristics.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    tld = "." + host.split(".")[-1] if "." in host else ""
    feats = {
        "host": host,
        "path": path,
        "tld": tld,
        "has_ip_host": bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)),
        "many_subdomains": host.count(".") >= 3,
        "has_at_symbol": "@" in url,
        "long_url": len(url) > 120,
        "suspicious_tld": tld in SUSPICIOUS_TLDS,
    }
    return feats

def heuristic_url_score(feats: Dict[str, Any]) -> float:
    """Convert URL features to a heuristic risk score (0-0.95)."""
    score = 0.0
    if feats.get("has_ip_host"): score += 0.3
    if feats.get("many_subdomains"): score += 0.2
    if feats.get("has_at_symbol"): score += 0.2
    if feats.get("long_url"): score += 0.1
    if feats.get("suspicious_tld"): score += 0.3
    if "login" in feats.get("path", "") or "verify" in feats.get("path", ""):
        score += 0.2
    return min(score, 0.95)

# -------------------------
# >>> Fusion of scores (NLP + Deep + Heur + Gemini)
# -------------------------
def fuse_scores(nlp: float, deep: float, heur: float, gem: Optional[float]) -> float:
    """
    Weighted fusion:
      base = 0.35*nlp + 0.35*deep + 0.30*heur
      If Gemini available, average with Gemini (50/50).
    """
    base = 0.35*nlp + 0.35*deep + 0.30*heur
    if gem is not None:
        try:
            return float(0.5*gem + 0.5*base)
        except Exception:
            return float(base)
    return float(base)

# -------------------------
# >>> Gemini helpers (optional)
# -------------------------
def get_gemini_client():
    """
    Return a genai Client instance. This is isolated so we only construct client when needed.
    """
    if not _HAS_GENAI:
        raise RuntimeError("google-genai SDK not available in environment")
    return genai.Client(api_key=GEMINI_API_KEY)

async def gemini_link_decision(url: str, context_text: Optional[str], page_fetch: bool = True) -> Dict[str, Any]:
    """
    Ask Gemini to classify a URL. Returns parsed JSON {risk_score,label,reasons,suggestions}.
    If Gemini not configured, raise HTTPException so callers gracefully degrade.
    """
    if not GEMINI_API_KEY or not _HAS_GENAI:
        raise HTTPException(status_code=400, detail="Gemini not configured")
    client = get_gemini_client()

    prompt = f"""
You are a security classifier for UPI/phishing links.
Return strict JSON with: risk_score (0-1), label (safe|suspicious|malicious), reasons[], suggestions[].
URL: {url}
Context: {context_text or ""}
"""
    cfg = gtypes.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema={
            "type":"object",
            "properties":{
                "risk_score":{"type":"number"},
                "label":{"type":"string"},
                "reasons":{"type":"array","items":{"type":"string"}},
                "suggestions":{"type":"array","items":{"type":"string"}}
            },
            "required":["risk_score","label","reasons","suggestions"]
        }
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(prompt), gtypes.Part.from_text(url)])],
        config=cfg
    )
    try:
        return resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text)
    except Exception:
        return {"risk_score": 0.5, "label": "suspicious", "reasons": ["parse_error"], "suggestions": ["retry"]}

async def gemini_message_decision(message: str, sender: Optional[str]) -> Dict[str, Any]:
    """Ask Gemini to classify a message. Same JSON schema as link decision."""
    if not GEMINI_API_KEY or not _HAS_GENAI:
        raise HTTPException(status_code=400, detail="Gemini not configured")
    client = get_gemini_client()

    prompt = f"""
Classify SMS/chat for UPI scam risk.
Return JSON: risk_score(0-1), label(safe|suspicious|malicious), reasons[], suggestions[].
Sender: {sender or "unknown"}
Message: {message}
"""
    cfg = gtypes.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema={
            "type":"object",
            "properties":{
                "risk_score":{"type":"number"},
                "label":{"type":"string"},
                "reasons":{"type":"array","items":{"type":"string"}},
                "suggestions":{"type":"array","items":{"type":"string"}}
            },
            "required":["risk_score","label","reasons","suggestions"]
        }
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(prompt)])],
        config=cfg
    )
    try:
        return resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text)
    except Exception:
        return {"risk_score": 0.5, "label": "suspicious", "reasons": ["parse_error"], "suggestions": ["retry"]}

# -------------------------
# >>> QR decoding helpers (pyzbar first, OpenCV fallback)
# -------------------------
def decode_qr_with_pyzbar(image: Image.Image) -> Optional[List[str]]:
    """
    Try decoding QR codes using pyzbar (native zbar). Returns list of decoded texts or None.
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except Exception:
        return None
    try:
        decoded = zbar_decode(image)
        if not decoded:
            return None
        return [d.data.decode("utf-8", errors="ignore") for d in decoded]
    except Exception:
        return None

def decode_qr_with_opencv(image: Image.Image) -> Optional[List[str]]:
    """
    Fallback: use OpenCV's QRCodeDetector (pure Python wheel). Returns list or None.
    """
    try:
        import cv2
        import numpy as _np
    except Exception:
        return None
    try:
        img = _np.array(image.convert("RGB"))
        # OpenCV uses BGR but detector works on RGB array too; convert for consistency
        detector = cv2.QRCodeDetector()
        # try multi decode (newer versions)
        try:
            ok, decoded_info, points = detector.detectAndDecodeMulti(img)
            if ok and decoded_info:
                return [d for d in decoded_info if d]
        except Exception:
            pass
        # fallback to single decode
        data, points, _ = detector.detectAndDecode(img)
        if data:
            return [data]
        return None
    except Exception:
        return None

def decode_qr_image_bytes(data: bytes) -> Optional[List[str]]:
    """Given raw image bytes, try pyzbar then OpenCV to extract QR payloads."""
    try:
        img = Image.open(BytesIO(data))
    except Exception:
        return None
    # try pyzbar first
    res = decode_qr_with_pyzbar(img)
    if res:
        return res
    # fallback to OpenCV
    return decode_qr_with_opencv(img)

# -------------------------
# >>> FastAPI app + Schemas
# -------------------------
app = FastAPI(title="FraudBlocker AI", version="1.2")

# Request/response Pydantic models
class InspectLinkRequest(BaseModel):
    url: str
    context_text: Optional[str] = None

class InspectMessageRequest(BaseModel):
    message: str
    sender: Optional[str] = None

class InspectQRRequest(BaseModel):
    qr_text: str

class InspectEmailRequest(BaseModel):
    raw_email: str

class RiskResult(BaseModel):
    risk_score: float
    label: str
    reasons: List[str]
    suggestions: List[str]
    gemini_decision: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = {}

# -------------------------
# >>> Endpoint: Inspect Link
# -------------------------
@app.post("/inspect/link", response_model=RiskResult)
async def inspect_link(req: InspectLinkRequest):
    # 1) extract URL features and heuristics
    feats = extract_url_features(req.url)
    heur = heuristic_url_score(feats)

    # 2) combine context and url for model inputs
    text_for_models = f"{req.context_text or ''} {req.url}"

    # 3) local ML/NLP scores
    nlp = nlp_score(text_for_models)
    deep = deep_score(text_for_models)

    # 4) optional Gemini decision (best-effort; degrade if not configured)
    try:
        gem = await gemini_link_decision(req.url, req.context_text, page_fetch=True) if GEMINI_API_KEY else None
        g_score = float(gem.get("risk_score")) if isinstance(gem, dict) and gem.get("risk_score") is not None else None
    except HTTPException:
        gem, g_score = None, None

    # 5) fused risk
    fused = fuse_scores(nlp, deep, heur, g_score)
    label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")

    # 6) reasons + suggestions assembly
    reasons = []
    if feats.get("suspicious_tld"): reasons.append("Suspicious TLD")
    if feats.get("has_ip_host"): reasons.append("IP host")
    if feats.get("many_subdomains"): reasons.append("Many subdomains")
    if feats.get("has_at_symbol"): reasons.append("Contains @")
    if "verify" in feats.get("path", ""): reasons.append("Verify in path")
    # append gem reasons if available
    if gem and isinstance(gem, dict):
        reasons += gem.get("reasons", [])

    suggestions = (gem.get("suggestions", []) if gem else []) or [
        "Avoid entering UPI PIN/OTP on web pages",
        "Verify payee UPI ID in official app",
        "Open links only from trusted sources"
    ]

    return RiskResult(
        risk_score=round(fused, 3),
        label=label,
        reasons=list(dict.fromkeys(reasons)),  # dedupe while preserving order
        suggestions=suggestions,
        gemini_decision=gem,
        artifacts={"features": feats, "url_sha256": sha256(req.url)}
    )

# -------------------------
# >>> Endpoint: Inspect Message (SMS/Chat)
# -------------------------
@app.post("/inspect/message", response_model=RiskResult)
async def inspect_message(req: InspectMessageRequest):
    text = req.message.strip()
    nlp = nlp_score(text)
    deep = deep_score(text)
    heur = min(sum(k in text.lower() for k in PHISH_KEYWORDS) * 0.1, 0.9)

    # Gemini (optional)
    try:
        gem = await gemini_message_decision(text, req.sender) if GEMINI_API_KEY else None
        g_score = float(gem.get("risk_score")) if isinstance(gem, dict) and gem.get("risk_score") is not None else None
    except HTTPException:
        gem, g_score = None, None

    fused = fuse_scores(nlp, deep, heur, g_score)
    label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")

    return RiskResult(
        risk_score=round(fused, 3),
        label=label,
        reasons=(gem.get("reasons", []) if gem else []) or [],
        suggestions=(gem.get("suggestions", []) if gem else []) or [
            "Never share OTP/UPI PIN",
            "Verify with official support",
            "Avoid tapping unknown links"
        ],
        gemini_decision=gem,
        artifacts={"message_sha256": sha256(text)}
    )

# -------------------------
# >>> Endpoint: Inspect QR (text payload)
# -------------------------
@app.post("/inspect/qr", response_model=RiskResult)
async def inspect_qr(req: InspectQRRequest):
    qr = req.qr_text.strip()
    looks_like_url = bool(re.match(r"^\w+://", qr) or qr.lower().startswith("upi:"))
    feats = extract_url_features(qr) if looks_like_url else {"path": "", "suspicious_tld": False, "has_ip_host": False, "many_subdomains": False, "has_at_symbol": False}
    heur = heuristic_url_score(feats) if looks_like_url else 0.1

    nlp = nlp_score(qr)
    deep = deep_score(qr)

    try:
        gem = await gemini_link_decision(qr, "QR decoded content", page_fetch=False) if GEMINI_API_KEY else None
        g_score = float(gem.get("risk_score")) if isinstance(gem, dict) and gem.get("risk_score") is not None else None
    except HTTPException:
        gem, g_score = None, None

    fused = fuse_scores(nlp, deep, heur, g_score)
    label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")

    return RiskResult(
        risk_score=round(fused, 3),
        label=label,
        reasons=(gem.get("reasons", []) if gem else []) or ([] if looks_like_url else ["No URL detected in QR"]),
        suggestions=(gem.get("suggestions", []) if gem else []) or [
            "Confirm payee in official UPI app",
            "Avoid scanning unknown QR codes",
            "Disable auto-approve of collect requests"
        ],
        gemini_decision=gem,
        artifacts={"qr_sha256": sha256(qr), "looks_like_url": looks_like_url}
    )

# -------------------------
# >>> Endpoint: Inspect QR Image (file upload)
# -------------------------
@app.post("/inspect/qr-image", response_model=RiskResult)
async def inspect_qr_image(file: UploadFile = File(...), context_text: Optional[str] = Form(None)):
    """
    Accept image file, decode the first QR payload found (pyzbar -> opencv fallback),
    then reuse the /inspect/qr processing pipeline logic to produce result.
    """
    data = await file.read()
    decoded_list = decode_qr_image_bytes(data)
    if not decoded_list:
        raise HTTPException(status_code=400, detail="No QR code found (tried pyzbar and OpenCV).")
    qr_payload = decoded_list[0]  # analyze the first payload by default

    # reuse same logic as text QR
    looks_like_url = bool(re.match(r"^\w+://", qr_payload) or qr_payload.lower().startswith("upi:"))
    feats = extract_url_features(qr_payload) if looks_like_url else {"path": "", "suspicious_tld": False, "has_ip_host": False, "many_subdomains": False, "has_at_symbol": False}
    heur = heuristic_url_score(feats) if looks_like_url else 0.1
    nlp = nlp_score(qr_payload)
    deep = deep_score(qr_payload)

    try:
        gem = await gemini_link_decision(qr_payload, context_text or "QR decoded content", page_fetch=False) if GEMINI_API_KEY else None
        g_score = float(gem.get("risk_score")) if isinstance(gem, dict) and gem.get("risk_score") is not None else None
    except HTTPException:
        gem, g_score = None, None

    fused = fuse_scores(nlp, deep, heur, g_score)
    label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")

    return RiskResult(
        risk_score=round(fused, 3),
        label=label,
        reasons=(gem.get("reasons", []) if gem else []) or ([] if looks_like_url else ["No URL detected in QR"]),
        suggestions=(gem.get("suggestions", []) if gem else []) or [
            "Confirm payee in official UPI app",
            "Avoid scanning unknown QR codes",
            "Disable auto-approve of collect requests"
        ],
        gemini_decision=gem,
        artifacts={"qr_sha256": sha256(qr_payload), "looks_like_url": looks_like_url, "decoded_text": qr_payload[:500]}
    )

# -------------------------
# >>> Endpoint: Inspect Email (raw RFC822)
# -------------------------
@app.post("/inspect/email", response_model=RiskResult)
async def inspect_email(req: InspectEmailRequest):
    """
    Accepts a raw RFC822 email string. Parses headers, attachments, and body
    to compute heuristics and run model/Gemini fusion.
    """
    raw = req.raw_email
    # parse raw email
    try:
        msg = message_from_string(raw, policy=policy.default)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid raw email format")

    # extract header signals
    from_hdr = msg.get("From", "")
    reply_to = msg.get("Reply-To", "")
    return_path = msg.get("Return-Path", "")
    subject = msg.get("Subject", "")
    received = msg.get_all("Received", []) or []

    # helper to extract domain from an email address
    def extract_domain(addr: str) -> str:
        m = re.search(r"@([A-Za-z0-9\.\-]+)", addr or "")
        return m.group(1).lower() if m else ""

    from_domain = extract_domain(from_hdr)
    reply_domain = extract_domain(reply_to)
    return_domain = extract_domain(return_path)

    reasons = []
    heur_score = 0.0

    # heuristic: urgent-sounding subject
    if any(k in (subject or "").lower() for k in ["urgent", "verify", "account", "suspend", "password", "otp"]):
        heur_score += 0.2
        reasons.append("Urgent-sounding subject")

    # reply-to mismatch heuristic
    if reply_to and reply_domain and from_domain and reply_domain != from_domain:
        heur_score += 0.25
        reasons.append("Reply-To domain differs from From domain")

    # return-path mismatch heuristic
    if return_path and return_domain and from_domain and return_domain != from_domain:
        heur_score += 0.15
        reasons.append("Return-Path domain differs from From domain")

    # attachments: scan attachment headers for executable types
    attachments = []
    for part in msg.iter_attachments():
        fn = part.get_filename()
        ctype = part.get_content_type()
        attachments.append({"filename": fn, "content_type": ctype})
        if ctype in ("application/x-msdownload", "application/x-msdos-program") or (fn and fn.lower().endswith((".exe", ".scr", ".bat", ".cmd", ".js"))):
            heur_score += 0.4
            reasons.append("Executable attachment")

    # body extraction (prefer text/plain)
    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    body_text += part.get_content()
                except Exception:
                    pass
    else:
        try:
            body_text = msg.get_content()
        except Exception:
            body_text = ""

    text_for_models = f"{subject or ''}\n{body_text or ''}\n{from_hdr or ''}"

    # models + link scanning inside body
    nlp = nlp_score(text_for_models)
    deep = deep_score(text_for_models)

    urls = re.findall(r"https?://[^\s'\"<>]+", body_text or "")
    link_heur = 0.0
    for u in urls:
        feats = extract_url_features(u)
        link_heur = max(link_heur, heuristic_url_score(feats))
        if feats.get("suspicious_tld"):
            reasons.append(f"Suspicious link TLD: {feats.get('tld')}")

    heur_score = min(heur_score + link_heur, 0.95)

    # optional Gemini message classification
    try:
        gem = await gemini_message_decision(text_for_models, sender=from_hdr) if GEMINI_API_KEY else None
        g_score = float(gem.get("risk_score")) if isinstance(gem, dict) and gem.get("risk_score") is not None else None
    except HTTPException:
        gem, g_score = None, None

    fused = fuse_scores(nlp, deep, heur_score, g_score)
    label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")

    suggestions = (gem.get("suggestions", []) if gem else []) or [
        "Do not click suspicious links",
        "Verify sender via a different channel",
        "Do not open unexpected attachments"
    ]

    return RiskResult(
        risk_score=round(fused, 3),
        label=label,
        reasons=list(dict.fromkeys(reasons + (gem.get("reasons", []) if gem else []))),
        suggestions=suggestions,
        gemini_decision=gem,
        artifacts={
            "from": from_hdr,
            "from_domain": from_domain,
            "reply_to": reply_to,
            "return_path": return_path,
            "urls_found": urls,
            "attachments": attachments,
            "message_sha256": sha256(raw)
        }
    )

# -------------------------
# >>> Health check
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
# -------------------------
# >>> Simple homepage + favicon to avoid 404 noise
# -------------------------
from fastapi.responses import HTMLResponse, Response

@app.get("/", response_class=HTMLResponse)
async def root():
    html = """
    <html>
      <head><title>FraudBlocker AI</title></head>
      <body style="font-family: Arial, sans-serif; max-width:800px;margin:40px;">
        <h1>FraudBlocker AI</h1>
        <p>API server is running. Use the OpenAPI docs to try endpoints:</p>
        <ul>
          <li><a href="/docs">Interactive Swagger docs (/docs)</a></li>
          <li><a href="/redoc">ReDoc documentation (/redoc)</a></li>
        </ul>
        <p>Endpoints: <code>/inspect/link</code>, <code>/inspect/message</code>, <code>/inspect/qr</code>, <code>/inspect/qr-image</code>, <code>/inspect/email</code></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

@app.get("/favicon.ico")
async def favicon():
    # return a 1x1 transparent PNG (prevents browser 404s). This is a tiny byte string.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00"
        b"\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(content=png_bytes, media_type="image/png")

# -------------------------
# >>> Run server (if run directly)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)



# import os
# import re
# import json
# import uvicorn
# import hashlib
# import socket
# from datetime import datetime, timezone
# from typing import List, Optional, Dict, Any
# from urllib.parse import urlparse, parse_qs
#
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
#
# # -------------------------------
# # CONFIG: Single place for API key
# # -------------------------------
# # Replace with your Gemini key, or set env var GEMINI_API_KEY
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAu64ChaCdgg--udtmoOsj00Or2H6u8Wis")  # <-- REPLACE KEY HERE
# GEMINI_MODEL = "gemini-2.5-flash"
#
# # -------------------------------
# # Gemini client (single instance)
# # -------------------------------
# from google import genai
# from google.genai import types as gtypes
#
# def get_gemini_client():
#     # Picks key from this one location or from env var as supported by SDK [uses single key]
#     return genai.Client(api_key=GEMINI_API_KEY)
#
# # ----------------------------------------
# # NLP + Deep Learning: minimal baseline
# # ----------------------------------------
# # NLP section: tokenization, TF-IDF features (classical NLP)
# # Deep learning section: lightweight neural model via Keras for binary classification
# # Both are optional baselines that run locally and fuse with Gemini's judgment.
#
# # ---- NLP (classical): TF-IDF + Logistic Regression ----
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
#
# NLP_SEED_SUSPICIOUS = [
#     "urgent your account will be blocked verify now",
#     "win lottery claim reward click link",
#     "update kyc immediately to avoid suspension",
#     "refund pending complete payment to receive",
#     "upi request from unknown sender approve now",
#     "security alert unusual login provide otp",
#     "investment double money guaranteed",
#     "official notice pay service fee immediately",
#     "free gift limited time click here",
#     "scam attempt payment link phishing"
# ]
# NLP_SEED_SAFE = [
#     "payment received successfully thank you",
#     "your order has been dispatched",
#     "upi payment to trusted merchant completed",
#     "monthly newsletter updates and offers",
#     "meeting at 3pm see you",
#     "your phone bill receipt is attached",
# ]
#
# VEC = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=6000)
# X = VEC.fit_transform(NLP_SEED_SUSPICIOUS + NLP_SEED_SAFE)
# y = [1]*len(NLP_SEED_SUSPICIOUS) + [0]*len(NLP_SEED_SAFE)
# NLP_CLS = LogisticRegression(max_iter=1000).fit(X, y)
#
# def nlp_score(text: str) -> float:
#     # NLP usage: transforms text via TF-IDF and predicts suspicious probability
#     vec = VEC.transform([text.lower()])
#     return float(NLP_CLS.predict_proba(vec)[0][1])
#
# # ---- Deep Learning: Tiny Keras model over simple features ----
# # Note: This is intentionally lightweight for single-file demo purposes.
# # In production, replace with a proper transformer or a trained model artifact.
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
#
# def make_deep_model(input_dim: int) -> keras.Model:
#     model = keras.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(16, activation="relu"),
#         layers.Dense(8, activation="relu"),
#         layers.Dense(1, activation="sigmoid"),
#     ])
#     model.compile(optimizer="adam", loss="binary_crossentropy")
#     return model
#
# # Build a simple URL/message feature extractor for the deep model
# PHISH_KEYWORDS = [
#     "kyc","update","verify","urgent","immediately","otp","limited","refund","reward",
#     "investment","double","gift","lottery","block","suspend","approve","verification"
# ]
# def dl_features(text: str) -> np.ndarray:
#     text_l = text.lower()
#     feats = [
#         len(text_l) / 500.0,                                  # normalized length
#         sum(k in text_l for k in PHISH_KEYWORDS) / 10.0,      # keyword density
#         text_l.count("http") / 3.0,                           # link count heuristic
#         int(any(c.isdigit() for c in text_l)) * 0.3,          # has digits (could be OTP/amount)
#         int("upi" in text_l) * 0.4,                           # mentions UPI
#     ]
#     return np.array(feats, dtype="float32")[None, ...]
#
# DL_INPUT_DIM = 5
# DEEP_MODEL = make_deep_model(DL_INPUT_DIM)
#
# # Quick synthetic training so the model produces non-trivial outputs
# X_train = []
# y_train = []
# for t in NLP_SEED_SUSPICIOUS + NLP_SEED_SAFE:
#     X_train.append(dl_features(t)[0])
#     y_train.append(1 if t in NLP_SEED_SUSPICIOUS else 0)
# X_train = np.array(X_train, dtype="float32")
# y_train = np.array(y_train, dtype="float32")
# DEEP_MODEL.fit(X_train, y_train, epochs=10, verbose=0)
#
# def deep_score(text: str) -> float:
#     # Deep learning usage: Keras model over engineered features
#     p = float(DEEP_MODEL.predict(dl_features(text), verbose=0)[0][0])
#     return p
#
# # ----------------------------------------
# # Heuristics for link inspection
# # ----------------------------------------
# SUSPICIOUS_TLDS = {".ru", ".tk", ".cn", ".ml", ".ga", ".gq"}
#
# def extract_url_features(url: str) -> Dict[str, Any]:
#     parsed = urlparse(url)
#     host = parsed.netloc.lower()
#     path = parsed.path.lower()
#     tld = "." + host.split(".")[-1] if "." in host else ""
#     feats = {
#         "host": host,
#         "path": path,
#         "tld": tld,
#         "has_ip_host": bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)),
#         "many_subdomains": host.count(".") >= 3,
#         "has_at_symbol": "@" in url,
#         "long_url": len(url) > 120,
#         "suspicious_tld": tld in SUSPICIOUS_TLDS,
#     }
#     return feats
#
# def heuristic_url_score(feats: Dict[str, Any]) -> float:
#     score = 0.0
#     if feats["has_ip_host"]: score += 0.3
#     if feats["many_subdomains"]: score += 0.2
#     if feats["has_at_symbol"]: score += 0.2
#     if feats["long_url"]: score += 0.1
#     if feats["suspicious_tld"]: score += 0.3
#     if "login" in feats["path"] or "verify" in feats["path"]: score += 0.2
#     return min(score, 0.95)
#
# def fuse_scores(nlp: float, deep: float, heur: float, gem: Optional[float]) -> float:
#     # Weighted fusion: prioritize Gemini if available
#     base = 0.35*nlp + 0.35*deep + 0.30*heur
#     if gem is not None:
#         return float(0.5*gem + 0.5*base)
#     return float(base)
#
# def sha256(s: str) -> str:
#     import hashlib
#     return hashlib.sha256(s.encode("utf-8")).hexdigest()
#
# # ----------------------------------------
# # FastAPI app and schemas
# # ----------------------------------------
# app = FastAPI(title="FraudBlocker AI", version="1.1")
#
# class InspectLinkRequest(BaseModel):
#     url: str
#     context_text: Optional[str] = None
#
# class InspectMessageRequest(BaseModel):
#     message: str
#     sender: Optional[str] = None
#
# class InspectQRRequest(BaseModel):
#     qr_text: str
#
# class RiskResult(BaseModel):
#     risk_score: float
#     label: str
#     reasons: List[str]
#     suggestions: List[str]
#     gemini_decision: Optional[Dict[str, Any]] = None
#     artifacts: Dict[str, Any] = {}
#
# # ----------------------------------------
# # Gemini usage helpers (clearly marked)
# # ----------------------------------------
# # GEMINI USAGE: URL Context + structured JSON output
# async def gemini_link_decision(url: str, context_text: Optional[str], page_fetch: bool = True) -> Dict[str, Any]:
#     if not GEMINI_API_KEY or GEMINI_API_KEY == "REPLACE_WITH_YOUR_GEMINI_API_KEY":
#         raise HTTPException(status_code=400, detail="Gemini API key not set")
#     client = get_gemini_client()
#
#     prompt = f"""
# You are a security classifier for UPI/phishing links.
# Return strict JSON with: risk_score (0-1), label (safe|suspicious|malicious), reasons[], suggestions[].
# URL: {url}
# Context: {context_text or ""}
# """
#     tools = [gtypes.Tool(url_context=gtypes.UrlContext())] if page_fetch else []
#     cfg = gtypes.GenerateContentConfig(
#         temperature=0.2,
#         tools=tools,
#         response_mime_type="application/json",
#         response_schema={
#             "type":"object",
#             "properties":{
#                 "risk_score":{"type":"number"},
#                 "label":{"type":"string"},
#                 "reasons":{"type":"array","items":{"type":"string"}},
#                 "suggestions":{"type":"array","items":{"type":"string"}}
#             },
#             "required":["risk_score","label","reasons","suggestions"]
#         }
#     )
#     resp = client.models.generate_content(
#         model=GEMINI_MODEL,
#         contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(prompt), gtypes.Part.from_text(url)])],
#         config=cfg
#     )
#     try:
#         return resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text)
#     except Exception:
#         return {"risk_score": 0.5, "label": "suspicious", "reasons": ["parse_error"], "suggestions": ["retry"]}
#
# # GEMINI USAGE: Message classification (text-only)
# async def gemini_message_decision(message: str, sender: Optional[str]) -> Dict[str, Any]:
#     if not GEMINI_API_KEY or GEMINI_API_KEY == "REPLACE_WITH_YOUR_GEMINI_API_KEY":
#         raise HTTPException(status_code=400, detail="Gemini API key not set")
#     client = get_gemini_client()
#     prompt = f"""
# Classify SMS/chat for UPI scam risk.
# Return JSON: risk_score(0-1), label(safe|suspicious|malicious), reasons[], suggestions[].
# Sender: {sender or "unknown"}
# Message: {message}
# """
#     cfg = gtypes.GenerateContentConfig(
#         temperature=0.2,
#         response_mime_type="application/json",
#         response_schema={
#             "type":"object",
#             "properties":{
#                 "risk_score":{"type":"number"},
#                 "label":{"type":"string"},
#                 "reasons":{"type":"array","items":{"type":"string"}},
#                 "suggestions":{"type":"array","items":{"type":"string"}}
#             },
#             "required":["risk_score","label","reasons","suggestions"]
#         }
#     )
#     resp = client.models.generate_content(
#         model=GEMINI_MODEL,
#         contents=[gtypes.Content(role="user", parts=[gtypes.Part.from_text(prompt)])],
#         config=cfg
#     )
#     try:
#         return resp.parsed if getattr(resp, "parsed", None) else json.loads(resp.text)
#     except Exception:
#         return {"risk_score": 0.5, "label": "suspicious", "reasons": ["parse_error"], "suggestions": ["retry"]}
#
# # ----------------------------------------
# # Endpoints
# # ----------------------------------------
# @app.post("/inspect/link", response_model=RiskResult)
# async def inspect_link(req: InspectLinkRequest):
#     feats = extract_url_features(req.url)
#     heur = heuristic_url_score(feats)
#     text_for_models = f"{req.context_text or ''} {req.url}"
#
#     # NLP usage
#     nlp = nlp_score(text_for_models)
#
#     # Deep learning usage
#     deep = deep_score(text_for_models)
#
#     # GEMINI usage (URL Context)
#     try:
#         gem = await gemini_link_decision(req.url, req.context_text, page_fetch=True)
#         g_score = float(gem.get("risk_score", None)) if isinstance(gem, dict) else None
#     except HTTPException:
#         gem, g_score = None, None
#
#     fused = fuse_scores(nlp, deep, heur, g_score)
#     label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")
#
#     reasons = []
#     if feats["suspicious_tld"]: reasons.append("Suspicious TLD")
#     if feats["has_ip_host"]: reasons.append("IP host")
#     if feats["many_subdomains"]: reasons.append("Many subdomains")
#     if feats["has_at_symbol"]: reasons.append("Contains @")
#     if "verify" in feats["path"]: reasons.append("Verify in path")
#
#     return RiskResult(
#         risk_score=round(fused, 3),
#         label=label,
#         reasons=reasons + (gem.get("reasons", []) if gem else []),
#         suggestions=(gem.get("suggestions", []) if gem else []) or [
#             "Avoid entering UPI PIN/OTP on web pages",
#             "Verify payee UPI ID in official app",
#             "Open links only from trusted sources"
#         ],
#         gemini_decision=gem,
#         artifacts={"features": feats, "url_sha256": sha256(req.url)}
#     )
#
# @app.post("/inspect/message", response_model=RiskResult)
# async def inspect_message(req: InspectMessageRequest):
#     text = req.message.strip()
#
#     # NLP usage
#     nlp = nlp_score(text)
#
#     # Deep learning usage
#     deep = deep_score(text)
#
#     # Heuristic: simple signal from keywords
#     heur = min(sum(k in text.lower() for k in PHISH_KEYWORDS) * 0.1, 0.9)
#
#     # GEMINI usage (text classification)
#     try:
#         gem = await gemini_message_decision(text, req.sender)
#         g_score = float(gem.get("risk_score", None)) if isinstance(gem, dict) else None
#     except HTTPException:
#         gem, g_score = None, None
#
#     fused = fuse_scores(nlp, deep, heur, g_score)
#     label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")
#
#     return RiskResult(
#         risk_score=round(fused, 3),
#         label=label,
#         reasons=(gem.get("reasons", []) if gem else []),
#         suggestions=(gem.get("suggestions", []) if gem else []) or [
#             "Never share OTP/UPI PIN",
#             "Verify with official support",
#             "Avoid tapping unknown links"
#         ],
#         gemini_decision=gem,
#         artifacts={"message_sha256": sha256(text)}
#     )
#
# @app.post("/inspect/qr", response_model=RiskResult)
# async def inspect_qr(req: InspectQRRequest):
#     qr = req.qr_text.strip()
#     # Treat QR content as text for NLP/Deep + heuristics if it looks like a URL
#     looks_like_url = bool(re.match(r"^\w+://", qr) or qr.lower().startswith("upi:"))
#     feats = extract_url_features(qr) if looks_like_url else {"path": "", "suspicious_tld": False, "has_ip_host": False, "many_subdomains": False, "has_at_symbol": False}
#     heur = heuristic_url_score(feats) if looks_like_url else 0.1
#
#     # NLP usage
#     nlp = nlp_score(qr)
#
#     # Deep learning usage
#     deep = deep_score(qr)
#
#     # GEMINI usage: analyze payload without forced fetch
#     try:
#         gem = await gemini_link_decision(qr, "QR decoded content", page_fetch=False)
#         g_score = float(gem.get("risk_score", None)) if isinstance(gem, dict) else None
#     except HTTPException:
#         gem, g_score = None, None
#
#     fused = fuse_scores(nlp, deep, heur, g_score)
#     label = "malicious" if fused >= 0.8 else ("suspicious" if fused >= 0.45 else "safe")
#
#     return RiskResult(
#         risk_score=round(fused, 3),
#         label=label,
#         reasons=(gem.get("reasons", []) if gem else []),
#         suggestions=(gem.get("suggestions", []) if gem else []) or [
#             "Confirm payee in official UPI app",
#             "Avoid scanning unknown QR codes",
#             "Disable auto-approve of collect requests"
#         ],
#         gemini_decision=gem,
#         artifacts={"qr_sha256": sha256(qr), "looks_like_url": looks_like_url}
#     )
#
# @app.get("/health")
# async def health():
#     return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
#
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
