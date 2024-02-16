#!flask/bin/python
import argparse
import io
import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import parse_qs

from flask import Flask, render_template, render_template_string, request, send_file
from TTS.config import load_config, load_speed_config_path
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--list_models", type=convert_boolean, nargs="?", const=True, default=False, help="list available pre-trained tts and vocoder models.")
    parser.add_argument("--model_name", type=str, default="tts_models/en/ljspeech/tacotron2-DDC", help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>")
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file.")
    parser.add_argument("--vocoder_path", type=str, help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).", default=None)
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser

args = create_argparser().parse_args()
path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path
##passing path so that we can read speed value from config file
    load_speed_config_path(args.config_path)
##    

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)
use_multi_speaker = (hasattr(synthesizer.tts_model, "num_speakers") and (
    synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
)) or (speaker_manager is not None)

language_manager = getattr(synthesizer.tts_model, "language_manager", None)
use_multi_language = (hasattr(synthesizer.tts_model, "num_languages") and (
    synthesizer.tts_model.num_languages > 1 or synthesizer.tts_languages_file is not None
)) or (language_manager is not None)

use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)

def find_default_style_wav(model_dir):
    """Find a default .wav file in the model directory to use for style transfer."""
    wav_files = list(Path(model_dir).glob("*.wav"))
    if wav_files:
        # Return the first .wav file found as the default style wav path
        return str(wav_files[0])
    else:
        # Return None if no .wav file is found
        return None

@app.route("/")
def index():
    return render_template("index.html", show_details=args.show_details, use_multi_speaker=use_multi_speaker, use_multi_language=use_multi_language, speaker_ids=speaker_manager.name_to_id if speaker_manager is not None else None, language_ids=language_manager.name_to_id if language_manager is not None else None, use_gst=use_gst)

@app.route("/details")
def details():
    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)
    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None
    return render_template("details.html", show_details=args.show_details, model_config=model_config, vocoder_config=vocoder_config, args=args.__dict__)

lock = Lock()

@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
        language_idx = request.headers.get("language-id") or request.values.get("language_id", "")
        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        print(f" > Model input: {text}")
        print(f" > Speaker Idx: {speaker_idx}")
        print(f" > Language Idx: {language_idx}")
        wavs = synthesizer.tts(text, language_name="en", style_wav=style_wav, speaker_wav=find_default_style_wav(os.path.dirname(model_path)))
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")

@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])

@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u")

# Original mary_tts_api_process() here
# @app.route("/process", methods=["GET", "POST"])
# def mary_tts_api_process():
    # with lock:
        # if request.method == "POST":
            # data = parse_qs(request.get_data(as_text=True))
            # text = data.get("INPUT_TEXT", [""])[0]
        # else:
            # text = request.args.get("INPUT_TEXT", "")
        # print(f" > Model input: {text}")
        # # Automatically find a default .wav file in the model directory
        # default_style_wav = find_default_style_wav(os.path.dirname(model_path))
        # if default_style_wav:
            # print(f"Using default style wav: {default_style_wav}")
            # wavs = synthesizer.tts(text, style_wav=default_style_wav)
        # else:
            # print(f"No default style wav found. Proceeding without it.")
            # wavs = synthesizer.tts(text)
        # out = io.BytesIO()
        # synthesizer.save_wav(wavs, out)
    # return send_file(out, mimetype="audio/wav")
#

#Modified mary_tts_api_process() here
@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
        print(f" > Model input: {text}")

        # Automatically find a default .wav file in the model directory
        default_style_wav = find_default_style_wav(os.path.dirname(model_path))


        if default_style_wav:
            print(f"Using default style wav: {default_style_wav}")
            # expects a parameter that indicates the path to a speaker's voice sample.
            wavs = synthesizer.tts(text, speaker_wav=default_style_wav)
        else:
            print(f"No default style wav found. Proceeding without it.")
            # Call the tts method without the speaker_wav parameter if no default style wav is found
            wavs = synthesizer.tts(text)

        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")
  

def main():
    app.run(debug=args.debug, host="::", port=args.port)

if __name__ == "__main__":
    main()
