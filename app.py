from flask import Flask, request, jsonify, render_template
import re

app = Flask(__name__, template_folder="templates")

# Mock database of known QR codes
KNOWN_QR_CODES = {
    "upi://pay?pa=user@upi": {
        "name": "John Doe",
        "upi_id": "user@upi",
        "type": "payment"
    },
    "https://example.com/user123": {
        "name": "Alice Smith",
        "url": "https://example.com/user123",
        "type": "profile"
    }
}

@app.route('/')
def home():
    return render_template("index.html")   # Serve frontend

@app.route('/process-qr', methods=['POST'])
def process_qr():
    data = request.json.get('data', '')

    result = KNOWN_QR_CODES.get(data, None)

    if result:
        return jsonify({"status": "success", "data": result})
    else:
        if data.startswith('upi://'):
            return jsonify({
                "status": "success",
                "data": {
                    "upi_id": extract_upi_id(data),
                    "type": "payment",

                    "message": "valid QR Code"
                }
            })
        return jsonify({"status": "error", "message": "Invalid QR Code"})

def extract_upi_id(upi_string):
    match = re.search(r'pa=([^&]+)', upi_string)
    return match.group(1) if match else upi_string

if __name__ == '__main__':
    app.run(debug=True)
