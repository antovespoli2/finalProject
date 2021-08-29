import os
import json
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import Levenshtein as Lev
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from optparse import OptionParser
from nisqa.NISQA_model import nisqaModel

def cer(s1, s2):
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

def wer(s1, s2):
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

def parse_metadata(filename):
    if not os.path.isfile(filename):
        return None

    with open(filename) as f:
        data = f.read().replace('\n', '')

    data = json.loads(data)

    must = ["generated_audio", "original_audio", "target_speaker"]

    for triple in data:
        if len(triple) > len(must) or not all(key in triple for key in must):
            return None

    return data

def compute_content(generated_audio, content_audio):
    logging.set_verbosity_error()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    audio_input, sample_rate = librosa.load(content_audio, sr = 16000)
    generated_input, sample_rate = librosa.load(generated_audio, sr = 16000)

    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values_generated = processor(generated_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    logits_generated = model(input_values_generated).logits
    predicted_ids_generated = torch.argmax(logits_generated, dim=-1)

    transcription = processor.decode(predicted_ids[0])
    transcription_generated = processor.decode(predicted_ids_generated[0])

    cer_res = cer(transcription, transcription_generated)
    wer_res = wer(transcription, transcription_generated)

    return cer_res, wer_res

def compute_style(generated_audio, target_style):
    fpath = Path(target_style)
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)

    fpath_generated = Path(generated_audio)
    wav_generated = preprocess_wav(fpath_generated)

    embed_generated = encoder.embed_utterance(wav_generated)

    return embed @ embed_generated

def compute_nmos(generated_audio)
    args = {'mode': 'predict_file', 'pretrained_model': 'NISQA/weights/nisqa_tts.tar', 'deg': generated_audio, 'data_dir': None, 'output_dir': None, 'csv_file': None, 'csv_deg': None, 'num_workers': None, 'bs': None}
    nisqa = nisqaModel(args)
    nmos = nisqa.predict_mod()
    
    return nmos

def compute_metrics(data, options):
    results = {key:[] for key in ["cer", "wer", "spk_sim", "nmos"]}

    if options["verbose"]:
        print("Computing metrics...")
        print()

    for i, datapoint in enumerate(data):
        if not options["metric"] or options["metric"] == "content":
            cer, wer = compute_content(datapoint["generated_audio"], datapoint["original_audio"])

            results["cer"].append(cer)
            results["wer"].append(wer)

            content_str = f"CER = {cer}, WER = {wer}, "

        else:
            content_str = ""

        if not options["metric"] or options["metric"] == "style":
            spk_sim = compute_style(datapoint["generated_audio"], datapoint["target_speaker"])

            results["spk_sim"].append(spk_sim)

            style_str = f"speaker_sim = {spk_sim}, "
        
        else:
            style_str = ""

        if not options["metric"] or options["metric"] == "nmos":
            nmos = compute_nmos(datapoint["generated_audio"])

            results["nmos"].append(nmos)

            nmos_str = f"NMOS = {nmos}"

        else:
            nmos_str = ""

        if options["verbose"]:
            print((f"[{i}]: " + content_str + style_str + nmos_str).strip(", "))

    results = {k:v for k,v in results.items() if v != []}

    if options["verbose"]:
        print()
        print()
    
    print("Final results:")

    if "cer" in results:
        print(f"    Avg CER: {results["cer"].mean()}")

    if "wer" in results:
        print(f"    Avg WER: {results["wer"].mean()}")

    if "spk_sim" in results:
        print(f"    Avg speaker sim: {results["spk_sim"].mean()}")

    if "nmos" in results:
        print(f"    Avg NMOS: {results["nmos"].mean()}")

    return results

def save_to_file(filename, results):
    with open(filename, "w") as f:
        f.write(results)

parser = OptionParser()
parser.add_option("-o", dest="filename",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-m",
                  action="store", type="string", dest="metric", default=None,
                  help="compute one specific metric")
parser.add_option("-v",
                  action="store_true", dest="verbose", default=False,
                  help="print all status messages to stdout")

(options, args) = parser.parse_args()

data = parse_metadata(args[0])

if len(args) != 1 or not data:
    parser.error("a valid metadata file must be provided")

results = compute_metrics(data, options)

if options["filename"]:
    save_to_file(filename, results)


