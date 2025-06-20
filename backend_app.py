from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import tempfile
import os
from elevenlabs import generate, set_api_key

# Load API keys (from environment variables ideally)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your-elevenlabs-api-key")

openai.api_key = OPENAI_API_KEY
set_api_key(ELEVENLABS_API_KEY)

app = Flask(__name__)
CORS(app)

last_audio_path = None

@app.route('/chat', methods=['POST'])
def chat():
    global last_audio_path
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"response": "Please send a message."}), 400

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        bot_response = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"response": f"OpenAI API error: {str(e)}"}), 500

    try:
        audio_bytes = generate(
            text=bot_response,
            voice="Rachel",
            model="eleven_monolingual_v1"
        )
        if last_audio_path and os.path.exists(last_audio_path):
            os.remove(last_audio_path)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_file.write(audio_bytes)
        tmp_file.close()
        last_audio_path = tmp_file.name
    except Exception as e:
        print(f"Audio generation error: {e}")
        last_audio_path = None

    return jsonify({
        "response": bot_response,
        "audio_url": "/audio" if last_audio_path else None
    })

@app.route('/audio')
def serve_audio():
    if last_audio_path and os.path.exists(last_audio_path):
        return send_file(last_audio_path, mimetype="audio/mpeg")
    else:
        return "No audio available", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
