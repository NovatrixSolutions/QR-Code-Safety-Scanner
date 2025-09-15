from flask import Flask, request, jsonify, render_template
import re
import hashlib
import time
from datetime import datetime
import requests
import json

app = Flask(__name__, template_folder="templates")

# Enhanced security database
THREAT_SIGNATURES = {
    'suspicious_domains': [
        'bit.ly', 'tinyurl.com', 't.co', 'short.link',
        'cutt.ly', 'is.gd', 'v.gd', 'ow.ly'
    ],
    'malicious_patterns': [
        r'[a-zA-Z0-9]{32}\.apk$',  # APK files
        r'javascript:',  # JavaScript execution
        r'data:text/html',  # Data URLs
        r'file://',  # Local file access
    ],
    'phishing_keywords': [
        'urgent', 'verify', 'suspended', 'click here',
        'congratulations', 'winner', 'claim now', 'limited time'
    ]
}

TRUSTED_DOMAINS = [
    'google.com', 'apple.com', 'microsoft.com',
    'paytm.com', 'phonepe.com', 'googlepay.com',
    'amazon.com', 'flipkart.com'
]

def analyze_qr_security(data):
    """Advanced security analysis of QR code data"""
    analysis = {
        'risk_level': 'low',
        'threats_detected': [],
        'recommendations': [],
        'technical_details': {},
        'confidence_score': 0
    }

    risk_score = 0

    # URL Analysis
    if data.startswith(('http://', 'https://')):
        analysis['technical_details']['type'] = 'URL'

        # Check for suspicious domains
        for domain in THREAT_SIGNATURES['suspicious_domains']:
            if domain in data:
                analysis['threats_detected'].append(f"URL Shortener Detected: {domain}")
                risk_score += 30

        # Check for trusted domains
        domain_trusted = any(trusted in data for trusted in TRUSTED_DOMAINS)
        if not domain_trusted:
            analysis['threats_detected'].append("Untrusted Domain")
            risk_score += 20

        # HTTPS check
        if not data.startswith('https://'):
            analysis['threats_detected'].append("Insecure HTTP Connection")
            risk_score += 25

        # Check for malicious patterns
        for pattern in THREAT_SIGNATURES['malicious_patterns']:
            if re.search(pattern, data, re.IGNORECASE):
                analysis['threats_detected'].append("Suspicious File Pattern Detected")
                risk_score += 40

    # UPI Analysis
    elif data.startswith('upi://'):
        analysis['technical_details']['type'] = 'UPI Payment'
        upi_id = extract_upi_id(data)
        analysis['technical_details']['upi_id'] = upi_id

        # Basic UPI validation
        if not re.match(r'^[a-zA-Z0-9.-]+@[a-zA-Z]+$', upi_id):
            analysis['threats_detected'].append("Invalid UPI ID Format")
            risk_score += 50

        # Check for suspicious amounts
        amount_match = re.search(r'am=([0-9.]+)', data)
        if amount_match:
            amount = float(amount_match.group(1))
            analysis['technical_details']['amount'] = amount
            if amount > 50000:  # Suspiciously high amount
                analysis['threats_detected'].append("High Transaction Amount")
                risk_score += 30

    # Text/Other Analysis
    else:
        analysis['technical_details']['type'] = 'Text/Other'

        # Check for phishing keywords
        for keyword in THREAT_SIGNATURES['phishing_keywords']:
            if keyword.lower() in data.lower():
                analysis['threats_detected'].append(f"Phishing Keyword: {keyword}")
                risk_score += 25

    # Determine risk level
    if risk_score >= 70:
        analysis['risk_level'] = 'high'
        analysis['recommendations'] = [
            "DO NOT scan this QR code",
            "Report to cybercrime authorities",
            "Block and delete immediately"
        ]
    elif risk_score >= 40:
        analysis['risk_level'] = 'medium'
        analysis['recommendations'] = [
            "Verify source before proceeding",
            "Check URL manually",
            "Use caution if proceeding"
        ]
    else:
        analysis['risk_level'] = 'low'
        analysis['recommendations'] = [
            "Appears safe to proceed",
            "Always verify payment details",
            "Keep security software updated"
        ]

    analysis['confidence_score'] = min(100, max(10, 100 - risk_score))
    analysis['technical_details']['risk_score'] = risk_score
    analysis['technical_details']['scan_timestamp'] = datetime.now().isoformat()

    return analysis

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/scanner')
def scanner():
    return render_template("scanner.html")

@app.route('/process-qr', methods=['POST'])
def process_qr():
    try:
        data = request.json.get('data', '')

        if not data:
            return jsonify({
                "status": "error",
                "message": "No QR code data received"
            })

        # Perform security analysis
        security_analysis = analyze_qr_security(data)

        # Basic information extraction
        basic_info = extract_basic_info(data)

        response = {
            "status": "success",
            "qr_data": data,
            "basic_info": basic_info,
            "security_analysis": security_analysis,
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        })

def extract_basic_info(data):
    """Extract basic information from QR code"""
    info = {'type': 'unknown', 'details': {}}

    if data.startswith('upi://'):
        info['type'] = 'UPI Payment'
        info['details'] = {
            'upi_id': extract_upi_id(data),
            'payment_app': 'UPI',
        }

        # Extract amount if present
        amount_match = re.search(r'am=([0-9.]+)', data)
        if amount_match:
            info['details']['amount'] = f"₹{amount_match.group(1)}"

    elif data.startswith(('http://', 'https://')):
        info['type'] = 'Website URL'
        info['details'] = {'url': data}

    elif data.startswith('tel:'):
        info['type'] = 'Phone Number'
        info['details'] = {'phone': data.replace('tel:', '')}

    elif data.startswith('mailto:'):
        info['type'] = 'Email'
        info['details'] = {'email': data.replace('mailto:', '')}

    else:
        info['type'] = 'Text Data'
        info['details'] = {'content': data[:100] + ('...' if len(data) > 100 else '')}

    return info

def extract_upi_id(upi_string):
    match = re.search(r'pa=([^&]+)', upi_string)
    return match.group(1) if match else upi_string

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# import re
#
# app = Flask(__name__, template_folder="templates")
#
# # Mock database of known QR codes
# KNOWN_QR_CODES = {
#     "upi://pay?pa=user@upi": {
#         "name": "John Doe",
#         "upi_id": "user@upi",
#         "type": "payment"
#     },
#     "https://example.com/user123": {
#         "name": "Alice Smith",
#         "url": "https://example.com/user123",
#         "type": "profile"
#     }
# }
#
# @app.route('/')
# def home():
#     return render_template("index.html")   # Serve frontend
#
# @app.route('/process-qr', methods=['POST'])
# def process_qr():
#     data = request.json.get('data', '')
#
#     result = KNOWN_QR_CODES.get(data, None)
#
#     if result:
#         return jsonify({"status": "success", "data": result})
#     else:
#         if data.startswith('upi://'):
#             return jsonify({
#                 "status": "success",
#                 "data": {
#                     "upi_id": extract_upi_id(data),
#                     "type": "payment",
#
#                     "message": "valid QR Code"
#                 }
#             })
#         return jsonify({"status": "error", "message": "Invalid QR Code"})
#
# def extract_upi_id(upi_string):
#     match = re.search(r'pa=([^&]+)', upi_string)
#     return match.group(1) if match else upi_string
#
# if __name__ == '__main__':
#     app.run(debug=True)
