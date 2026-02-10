from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import torch
import numpy as np
import time
import os
import torchaudio
import gradio as gr
import urllib.parse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pocket_tts import TTSModel
import scipy.io.wavfile
import soundfile as sf
from pyngrok import ngrok
import tempfile
import threading
#initialize the flask app 
app = Flask(__name__)
#create a variable to store ngrok key
NGROK_AUTH_TOKEN = "*****"
#define the homepage route
@app.route("/")
def index():
    return render_template("index.html")
# Define the /process route that handles POST requests with audio files
@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    #define a variable name that will store coverted WAV file path
    wav_path = None
    try:
        waveform, sr = torchaudio.load(tmp_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        # Save audio in clean WAV file
        wav_path = tmp_path + ".wav"
        torchaudio.save(wav_path, waveform, 16000)
        # Process the audio file to get transcription, AI response, and output audio path
        transcription, ai_text, audio_path = process_audio(wav_path)
        #create an audio url(audio file path) to save audio files
        audio_url = f"/static/audio/{os.path.basename(audio_path)}" if audio_path and os.path.exists(audio_path) else None
        #return json response for all processed data
        return jsonify({
            "transcription": transcription,
            "ai_response": ai_text,
            "audio_url": audio_url
        })

    except Exception as e:
        print("Process error:", str(e))
        return jsonify({"error": str(e)}), 200
    finally:
        for p in (tmp_path, wav_path):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except: pass
#define route to server generated audio files
@app.route("/static/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory("/content/static/audio", filename)

# Load models to the CPU first for effciency
#load whisper processor for processing audio data
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
#load whisper model for speech-to-text transcription
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
#load the text-to-speech model
tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt("alba")


# process_audio (unchanged, now always gets clean WAV)
def process_audio(audio_path):
    if audio_path is None:
        return "No audio detected", "N/A", None
    try:
        data, sr = sf.read(audio_path, dtype="float32")
        if len(data.shape) > 1: data = np.mean(data, axis=1)
        audio = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.squeeze().to(device)

        audio_np = audio.cpu().numpy()
        input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = model.generate(input_features, max_new_tokens=120)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        if not transcription:
            return "Transcription failed (empty)", "No AI response", None

        audio_text = urllib.parse.quote(transcription)
        resp = requests.get(f"https://gen.pollinations.ai/text/{audio_text}?key=*****", timeout=15)
        ai_text = resp.text.strip() if resp.status_code == 200 else f"API error {resp.status_code}"

        audio_tensor = tts_model.generate_audio(voice_state, ai_text)
        output_filename = f"/content/static/audio/ai_voiceresponse_{int(time.time())}.wav"
        scipy.io.wavfile.write(output_filename, tts_model.sample_rate, (audio_tensor.numpy() * 32767).astype(np.int16))

        return transcription, ai_text, output_filename
    except Exception as e:
        error_msg = f"Error: {type(e).__name__} – {str(e)[:200]}"
        print(error_msg)
        return error_msg, "N/A", None

def app_run():
    app.run(port=5001, debug=False, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=app_run, daemon=True).start()
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(5001)
    print("\n NGROK public url:", public_url)
