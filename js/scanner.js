/**
 * scanner.js
 * Unified frontend JS for index.html and scanner.html
 *
 * - connects to backend endpoints:
 *   POST /inspect/link        { url, context_text? }
 *   POST /inspect/message     { message, sender? }
 *   POST /inspect/email       { raw_email }
 *   POST /inspect/qr-image    multipart/form-data { file, context_text? }
 *
 * - Exposes global helpers (scanURL, scanMessage, scanEmail, scanQR)
 * - Auto-attaches to elements when present on the page
 *
 * Change API_BASE if your backend isn't on http://127.0.0.1:8000
 */

const API_BASE = "http://127.0.0.1:8000"; // update if needed

document.addEventListener("DOMContentLoaded", () => {
  // mobile menu toggle (works for both pages)
  const mobileMenu = document.getElementById("mobileMenu");
  const navLinks = document.querySelector(".nav-links");
  if (mobileMenu && navLinks) {
    mobileMenu.addEventListener("click", () => {
      navLinks.style.display = navLinks.style.display === "flex" ? "none" : "flex";
    });
  }

  // Wire up controls present on page (index or scanner)
  safeHook("#scanLink", setupLinkScan);
  safeHook("#linkInput", setupLinkScan); // scanner.html
  safeHook("#scanMessage", setupMessageScan);
  safeHook("#msgInput", setupMessageScan); // scanner.html
  safeHook("#scanEmail", setupEmailScan);
  safeHook("#emailInput", setupEmailScan); // scanner.html
  safeHook("#qrFile", setupQRUpload);
  safeHook("#qrImage", setupQRUpload); // scanner.html
});

/* Helper: if element exists, call setup function */
function safeHook(selector, setupFn) {
  const el = document.querySelector(selector);
  if (el) setupFn(el);
}

/* --------------------------
   LINK SCAN
   -------------------------- */
function setupLinkScan() {
  const input = document.querySelector("#linkInput") || document.querySelector("#scanLink");
  const resultElem = document.querySelector("#linkResult") || document.querySelector("#urlOutput");
  const contextInput = document.querySelector("#linkContext") || null;

  const btn = document.querySelector("#scanLink") || Array.from(document.querySelectorAll("button"))
    .find(b => /scan|analyz/i.test(b.textContent || "") && /link|url/i.test(b.parentElement?.textContent || b.textContent || ""));

  if (btn) btn.addEventListener("click", () => performLinkScan(input, contextInput, resultElem));
  // global helper
  window.scanURL = () => performLinkScan(input, contextInput, resultElem);
}

async function performLinkScan(input, contextInput, resultElem) {
  if (!input) return alert("Link input not found on page.");
  const url = (input.value || "").trim();
  const context_text = contextInput ? (contextInput.value || "").trim() : undefined;
  if (!url) return simpleRender(resultElem, { error: "Please enter a URL" });

  setLoading(resultElem, true);
  try {
    const res = await fetch(`${API_BASE}/inspect/link`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, context_text })
    });
    await handleResponseAndRender(res, resultElem);
  } catch (err) {
    simpleRender(resultElem, { error: err.message });
  } finally {
    setLoading(resultElem, false);
  }
}

/* --------------------------
   MESSAGE SCAN
   -------------------------- */
function setupMessageScan() {
  const input = document.querySelector("#msgInput") || document.querySelector("#scanMessage");
  const sender = document.querySelector("#msgSender") || null;
  const resultElem = document.querySelector("#msgResult") || document.querySelector("#messageOutput");

  const btn = document.querySelector("#scanMessage") || Array.from(document.querySelectorAll("button"))
    .find(b => /scan|analyz/i.test(b.textContent || "") && /message|sms|chat/i.test(b.parentElement?.textContent || b.textContent || ""));

  if (btn) btn.addEventListener("click", () => performMessageScan(input, sender, resultElem));
  window.scanMessage = () => performMessageScan(input, sender, resultElem);
}

async function performMessageScan(input, senderInput, resultElem) {
  if (!input) return alert("Message input not found.");
  const message = (input.value || "").trim();
  const sender = senderInput ? (senderInput.value || "").trim() : undefined;
  if (!message) return simpleRender(resultElem, { error: "Please enter message text" });

  setLoading(resultElem, true);
  try {
    const res = await fetch(`${API_BASE}/inspect/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, sender })
    });
    await handleResponseAndRender(res, resultElem);
  } catch (err) {
    simpleRender(resultElem, { error: err.message });
  } finally {
    setLoading(resultElem, false);
  }
}

/* --------------------------
   EMAIL SCAN
   -------------------------- */
function setupEmailScan() {
  const input = document.querySelector("#emailInput") || document.querySelector("#scanEmail");
  const resultElem = document.querySelector("#emailResult") || document.querySelector("#emailOutput");

  const btn = document.querySelector("#scanEmail") || Array.from(document.querySelectorAll("button"))
    .find(b => /scan|analyz/i.test(b.textContent || "") && /email/i.test(b.parentElement?.textContent || b.textContent || ""));

  if (btn) btn.addEventListener("click", () => performEmailScan(input, resultElem));
  window.scanEmail = () => performEmailScan(input, resultElem);
}

async function performEmailScan(input, resultElem) {
  if (!input) return alert("Email input not found.");
  const raw_email = (input.value || "").trim();
  if (!raw_email) return simpleRender(resultElem, { error: "Please paste the raw email content" });

  setLoading(resultElem, true);
  try {
    const res = await fetch(`${API_BASE}/inspect/email`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_email })
    });
    await handleResponseAndRender(res, resultElem);
  } catch (err) {
    simpleRender(resultElem, { error: err.message });
  } finally {
    setLoading(resultElem, false);
  }
}

/* --------------------------
   QR IMAGE UPLOAD & SCAN
   -------------------------- */
function setupQRUpload() {
  const input = document.querySelector("#qrImage") || document.querySelector("#qrFile");
  const contextInput = document.querySelector("#qrContext") || null;
  const resultElem = document.querySelector("#resultCard") || document.querySelector("#qrResult") || document.querySelector("#qrOutput");

  if (!input) return;
  // Attach analyze button if present
  const analyzeBtn = document.querySelector("#scanQR") || Array.from(document.querySelectorAll("button"))
    .find(b => /scan|analyz|upload/i.test(b.textContent || "") && /qr|image/i.test(b.parentElement?.textContent || b.textContent || ""));

  if (analyzeBtn) analyzeBtn.addEventListener("click", () => performQRUpload(input, contextInput, resultElem));
  // also perform when file changes (useful for quick testing)
  input.addEventListener("change", () => {
    const hasSeparateBtn = !!analyzeBtn;
    if (!hasSeparateBtn) performQRUpload(input, contextInput, resultElem);
  });

  window.scanQR = () => performQRUpload(input, contextInput, resultElem);
}

async function performQRUpload(input, contextInput, resultElem) {
  const file = input.files && input.files[0];
  if (!file) return simpleRender(resultElem, { error: "No file chosen" });

  // client-side validations
  const validTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];
  if (!validTypes.includes(file.type)) {
    return simpleRender(resultElem, { error: "Invalid file type. Use JPG/PNG/GIF/WEBP" });
  }
  if (file.size > 10 * 1024 * 1024) {
    return simpleRender(resultElem, { error: "File too large (max 10MB)" });
  }

  setLoading(resultElem, true);
  try {
    const form = new FormData();
    form.append("file", file);

    // optional context: find nearby context input if present
    if (contextInput && contextInput.value) form.append("context_text", contextInput.value);

    const res = await fetch(`${API_BASE}/inspect/qr-image`, {
      method: "POST",
      body: form
    });

    await handleResponseAndRender(res, resultElem);
  } catch (err) {
    simpleRender(resultElem, { error: err.message });
  } finally {
    setLoading(resultElem, false);
  }
}

/* --------------------------
   Response handling + rendering helpers
   -------------------------- */
async function handleResponseAndRender(res, resultElem) {
  if (!res) return simpleRender(resultElem, { error: "No response" });
  const contentType = res.headers.get("content-type") || "";
  let data;
  if (contentType.includes("application/json")) {
    data = await res.json();
  } else {
    const text = await res.text();
    // if backend returned non-json, show text
    data = { raw: text, status: res.status, ok: res.ok };
  }

  // If API returned an error (FastAPI style)
  if (!res.ok && data) {
    // show returned detail if present
    return simpleRender(resultElem, data);
  }

  prettyRenderResult(resultElem, data);
}

function setLoading(el, on) {
  if (!el) return;
  if (on) {
    // store previous content and show scanning indicator
    el.dataset._previous = el.innerHTML;
    if (el.tagName === "PRE") {
      el.textContent = "Scanning…";
    } else {
      el.innerHTML = "<em>Scanning…</em>";
    }
    el.style.opacity = 0.9;
  } else {
    el.style.opacity = 1;
  }
}

function simpleRender(el, obj) {
  if (!el) {
    console.log(obj);
    return;
  }
  if (el.tagName === "PRE") {
    el.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
    return;
  }
  el.innerHTML = `<pre style="white-space:pre-wrap">${escapeHtml(typeof obj === "string" ? obj : JSON.stringify(obj, null, 2))}</pre>`;
}

function prettyRenderResult(el, data) {
  // prefer a "card" style if the page has #resultCard
  const isCard = !!document.querySelector("#resultCard") && (el === document.querySelector("#resultCard") || el.id === "resultCard");

  if (!el) {
    console.log(data);
    return;
  }

  // handle FastAPI error shape
  if (data && data.detail && !("risk_score" in data)) {
    return simpleRender(el, data);
  }

  if (isCard && data && typeof data === "object") {
    const score = ("risk_score" in data) ? (data.risk_score) : (data.risk || data.score || null);
    const label = data.label || (score !== null ? (score >= 0.8 ? "malicious" : (score >= 0.45 ? "suspicious" : "safe")) : "unknown");
    const reasons = Array.isArray(data.reasons) ? data.reasons : [];
    const suggestions = Array.isArray(data.suggestions) ? data.suggestions : (data.suggestion ? [data.suggestion] : []);
    const detailsJson = JSON.stringify(data, null, 2);

    let html = `
      <div class="result-header" style="display:flex;justify-content:space-between;gap:1rem;align-items:start">
        <div class="result-status"><strong>Result:</strong> ${escapeHtml(String(label)).toUpperCase()}</div>
        <div class="result-confidence"><strong>Score:</strong> ${score !== null ? escapeHtml(String(score)) : "N/A"}</div>
      </div>
      <div class="result-body" style="margin-top:0.5rem">
        ${reasons.length ? `<div class="threat-indicators"><h4>Reasons</h4><ul>` + reasons.map(r => `<li>${escapeHtml(r)}</li>`).join("") + `</ul></div>` : ""}
        ${suggestions.length ? `<div class="recommendations"><h4>Suggestions</h4><ul>` + suggestions.map(s => `<li>${escapeHtml(s)}</li>`).join("") + `</ul></div>` : ""}
        <div class="technical-details"><h4>Raw (JSON)</h4><pre style="max-height:300px;overflow:auto">${escapeHtml(detailsJson)}</pre></div>
      </div>
    `;
    el.innerHTML = html;
    el.scrollIntoView({ behavior: "smooth" });
    return;
  }

  // fallback: pretty JSON in <pre>
  simpleRender(el, data);
}

/* small util */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}


























//
//
//
//
//// JavaScript for Scanner page
//
//document.addEventListener('DOMContentLoaded', function() {
//    // Mobile menu toggle
//    const mobileMenu = document.getElementById('mobileMenu');
//    const navLinks = document.querySelector('.nav-links');
//
//    if (mobileMenu) {
//        mobileMenu.addEventListener('click', function() {
//            navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
//        });
//    }
//
//    // Elements
//    const uploadArea = document.getElementById('uploadArea');
//    const qrInput = document.getElementById('qrInput');
//    const chooseFileBtn = document.getElementById('chooseFileBtn');
//    const imagePreview = document.getElementById('imagePreview');
//    const previewImg = document.getElementById('previewImg');
//    const fileName = document.getElementById('fileName');
//    const fileSize = document.getElementById('fileSize');
//    const clearImage = document.getElementById('clearImage');
//    const analyzeBtn = document.getElementById('analyzeBtn');
//    const resultsSection = document.getElementById('resultsSection');
//    const resultCard = document.getElementById('resultCard');
//
//    // Event Listeners
//    chooseFileBtn.addEventListener('click', () => qrInput.click());
//
//    qrInput.addEventListener('change', handleFileSelect);
//
//    uploadArea.addEventListener('dragover', (e) => {
//        e.preventDefault();
//        uploadArea.style.borderColor = 'var(--primary-color)';
//        uploadArea.style.backgroundColor = '#f8fafc';
//    });
//
//    uploadArea.addEventListener('dragleave', () => {
//        uploadArea.style.borderColor = 'var(--border-color)';
//        uploadArea.style.backgroundColor = 'white';
//    });
//
//    uploadArea.addEventListener('drop', (e) => {
//        e.preventDefault();
//        uploadArea.style.borderColor = 'var(--border-color)';
//        uploadArea.style.backgroundColor = 'white';
//
//        if (e.dataTransfer.files.length) {
//            qrInput.files = e.dataTransfer.files;
//            handleFileSelect();
//        }
//    });
//
//    clearImage.addEventListener('click', resetUpload);
//
//    analyzeBtn.addEventListener('click', analyzeQRCode);
//
//    // Functions
//    function handleFileSelect() {
//        if (!qrInput.files || !qrInput.files[0]) return;
//
//        const file = qrInput.files[0];
//        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
//
//        if (!validTypes.includes(file.type)) {
//            alert('Please select a valid image file (JPG, PNG, GIF)');
//            resetUpload();
//            return;
//        }
//
//        if (file.size > 10 * 1024 * 1024) {
//            alert('File size exceeds 10MB limit');
//            resetUpload();
//            return;
//        }
//
//        const reader = new FileReader();
//
//        reader.onload = function(e) {
//            previewImg.src = e.target.result;
//            fileName.textContent = file.name;
//            fileSize.textContent = formatFileSize(file.size);
//            imagePreview.style.display = 'block';
//            analyzeBtn.disabled = false;
//        };
//
//        reader.readAsDataURL(file);
//    }
//
//    function resetUpload() {
//        qrInput.value = '';
//        previewImg.src = '';
//        imagePreview.style.display = 'none';
//        analyzeBtn.disabled = true;
//        resultsSection.style.display = 'none';
//    }
//
//    function formatFileSize(bytes) {
//        if (bytes < 1024) return bytes + ' bytes';
//        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
//        else return (bytes / 1048576).toFixed(1) + ' MB';
//    }
//
//    function analyzeQRCode() {
//        // Show loading state
//        analyzeBtn.querySelector('.loading-spinner').style.display = 'block';
//        analyzeBtn.disabled = true;
//
//        // Simulate analysis (in a real app, this would be an API call)
//        setTimeout(() => {
//            analyzeBtn.querySelector('.loading-spinner').style.display = 'none';
//            analyzeBtn.disabled = false;
//
//            // Generate fake analysis results
//            const isSafe = Math.random() > 0.5;
//            const threatLevel = isSafe ? 'safe' : (Math.random() > 0.5 ? 'suspicious' : 'malicious');
//
//            displayResults(threatLevel);
//        }, 2000);
//    }
//
//    function displayResults(threatLevel) {
//        // Sample data for demonstration
//        const sampleData = {
//            safe: {
//                status: 'Safe',
//                icon: 'fa-check-circle',
//                class: 'status-safe',
//                confidence: '98%',
//                url: 'https://example.com/safe-redirect',
//                indicators: [
//                    { type: 'Domain Age', status: 'safe', text: 'Registered for 3+ years' },
//                    { type: 'SSL Certificate', status: 'safe', text: 'Valid HTTPS' },
//                    { type: 'Content Analysis', status: 'safe', text: 'No malicious content' },
//                    { type: 'Reputation', status: 'safe', text: 'Trusted website' }
//                ],
//                details: [
//                    { label: 'Domain Created', value: 'March 12, 2019' },
//                    { label: 'Server Location', value: 'United States' },
//                    { label: 'Redirects', value: 'None detected' },
//                    { label: 'Tracking', value: 'Minimal analytics' }
//                ],
//                recommendation: 'This QR code appears to be safe. You can proceed with caution.'
//            },
//            suspicious: {
//                status: 'Suspicious',
//                icon: 'fa-exclamation-triangle',
//                class: 'status-warning',
//                confidence: '65%',
//                url: 'http://suspicious-site.xyz/offer?uid=283742',
//                indicators: [
//                    { type: 'Domain Age', status: 'warning', text: 'Registered 15 days ago' },
//                    { type: 'SSL Certificate', status: 'danger', text: 'No HTTPS encryption' },
//                    { type: 'Content Analysis', status: 'safe', text: 'No malicious content' },
//                    { type: 'Reputation', status: 'warning', text: 'Unknown reputation' }
//                ],
//                details: [
//                    { label: 'Domain Created', value: 'July 5, 2025' },
//                    { label: 'Server Location', value: 'Offshore hosting' },
//                    { label: 'Redirects', value: '2 redirects detected' },
//                    { label: 'Tracking', value: 'Multiple tracking parameters' }
//                ],
//                recommendation: 'This QR code shows suspicious characteristics. Avoid entering any personal information.'
//            },
//            malicious: {
//                status: 'Malicious',
//                icon: 'fa-times-circle',
//                class: 'status-danger',
//                confidence: '92%',
//                url: 'http://malicious-site.biz/login.php?redirect=paypal.com',
//                indicators: [
//                    { type: 'Domain Age', status: 'danger', text: 'Registered 2 days ago' },
//                    { type: 'SSL Certificate', status: 'danger', text: 'No HTTPS encryption' },
//                    { type: 'Content Analysis', status: 'danger', text: 'Phishing page detected' },
//                    { type: 'Reputation', status: 'danger', text: 'Blacklisted domain' }
//                ],
//                details: [
//                    { label: 'Domain Created', value: 'August 18, 2025' },
//                    { label: 'Server Location', value: 'Unknown jurisdiction' },
//                    { label: 'Redirects', value: '3 redirects to suspicious domain' },
//                    { label: 'Tracking', value: 'Extensive user tracking' }
//                ],
//                recommendation: 'This QR code is highly likely to be malicious. Do not proceed and delete this QR code immediately.'
//            }
//        };
//
//        const data = sampleData[threatLevel];
//
//        // Build results HTML
//        let html = `
//            <div class="result-header">
//                <div class="result-status">
//                    <i class="fas ${data.icon} status-icon ${data.class}"></i>
//                    <span>${data.status}</span>
//                </div>
//                <div class="result-confidence">Confidence: ${data.confidence}</div>
//            </div>
//            <div class="result-body">
//                <a href="#" class="result-url">${data.url}</a>
//
//                <div class="threat-indicators">
//                    <h3>Security Indicators</h3>
//                    <div class="indicator-grid">
//        `;
//
//        data.indicators.forEach(indicator => {
//            html += `
//                <div class="indicator-item">
//                    <i class="fas ${indicator.status === 'safe' ? 'fa-check-circle indicator-safe' :
//                                  indicator.status === 'warning' ? 'fa-exclamation-triangle indicator-warning' :
//                                  'fa-times-circle indicator-danger'} indicator-icon"></i>
//                    <div>
//                        <strong>${indicator.type}:</strong> ${indicator.text}
//                    </div>
//                </div>
//            `;
//        });
//
//        html += `
//                    </div>
//                </div>
//
//                <div class="result-details">
//                    <h3>Detailed Analysis</h3>
//        `;
//
//        data.details.forEach(detail => {
//            html += `
//                <div class="detail-item">
//                    <span class="detail-label">${detail.label}</span>
//                    <span>${detail.value}</span>
//                </div>
//            `;
//        });
//
//        html += `
//                </div>
//
//                <div class="recommendation">
//                    <h3>Security Recommendation</h3>
//                    <p>${data.recommendation}</p>
//                </div>
//            </div>
//        `;
//
//        resultCard.innerHTML = html;
//        resultsSection.style.display = 'block';
//
//        // Scroll to results
//        resultsSection.scrollIntoView({ behavior: 'smooth' });
//    }
//});