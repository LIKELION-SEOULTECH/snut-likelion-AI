from flask import Flask, request, jsonify
from kobart_sum import KoBARTSummarizer

app = Flask(__name__)
summarizer = KoBARTSummarizer()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data.get("text", "")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        summary = summarizer.summarize(input_text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
