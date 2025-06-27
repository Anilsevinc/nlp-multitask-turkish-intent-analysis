from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import numpy as np
from rvc_python.infer import RVCInference
import os
import librosa
import soundfile as sf

# Paths and default values
tts_output_path = "tts_output.wav"
default_rvc_model_path = "models/FurinaEN/FurinaEN.pth"
default_converted_output_path = "converted_voice.wav"

# Load TTS model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

# Initialize RVC inference
rvc = RVCInference(device="cuda")

# Reset pipeline function
def reset_pipeline():
    """
    Resets the processing pipeline, clearing intermediate tensors.
    """
    global feats, feats0, rvc
    feats = None
    feats0 = None
    if hasattr(rvc, "clear_cache"):
        rvc.clear_cache()  # Clears RVC's internal cache if supported
    print("Pipeline and tensors have been reset.")

def get_available_models(folder="models"):
    """Returns a list of available RVC model files in the specified folder."""
    return [f for f in os.listdir(folder) if f.endswith(".pth")]

def preprocess_audio(input_path, output_path, target_sr):
    """
    Resamples and trims audio to prepare it for RVC processing.
    Args:
        input_path (str): Path to input audio file.
        output_path (str): Path to save the processed audio.
        target_sr (int): Target sampling rate.
    """
    audio, sr = librosa.load(input_path, sr=None)
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    sf.write(output_path, audio, target_sr)

def process_tts_and_rvc(text, rvc_model_path, pitch):
    """
    Processes text through TTS and converts it with RVC.
    Args:
        text (str): Input text for TTS.
        rvc_model_path (str): Path to the RVC model file.
        pitch (int): Pitch adjustment for voice conversion.
    Returns:
        str: Path to the converted voice output or error message.
    """
    try:
        # Step 1: TTS - Generate waveform from text
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        # Save TTS output as WAV
        waveform_np = output.squeeze().cpu().numpy()
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        scipy.io.wavfile.write(tts_output_path, rate=model.config.sampling_rate, data=waveform_int16)

        # Preprocess the TTS output
        processed_tts_path = "processed_tts_output.wav"
        preprocess_audio(tts_output_path, processed_tts_path, target_sr=model.config.sampling_rate)

        # Step 2: RVC - Load model and perform voice conversion
        rvc.load_model(rvc_model_path)
        rvc.set_params(
            f0method="harvest",
            f0up_key=pitch,
            index_rate=0.5,
            filter_radius=3,
            rms_mix_rate=0.25,
            protect=0.33
        )

        # Perform inference
        converted_output_path = default_converted_output_path
        rvc.infer_file(processed_tts_path, converted_output_path)

        # Reset pipeline after processing
        reset_pipeline()
        return converted_output_path

    except Exception as e:
        reset_pipeline()  # Ensure cleanup even in case of errors
        return f"Error during processing: {str(e)}"

def debug_tensors(tensor_a, tensor_b):
    """
    Aligns tensors by truncating them to the smallest common size if necessary.
    Args:
        tensor_a (torch.Tensor): First tensor.
        tensor_b (torch.Tensor): Second tensor.
    Returns:
        tuple: Aligned tensors.
    """
    if tensor_a.shape[1] != tensor_b.shape[1]:
        min_len = min(tensor_a.shape[1], tensor_b.shape[1])
        print(f"Aligning tensors: {tensor_a.shape[1]} -> {min_len}, {tensor_b.shape[1]} -> {min_len}")
        tensor_a = tensor_a[:, :min_len]
        tensor_b = tensor_b[:, :min_len]
    return tensor_a, tensor_b

# Example usage
if __name__ == "__main__":
    text = "Merhaba, bu bir test mesajıdır."
    selected_model = default_rvc_model_path  # Path to your desired RVC model
    pitch_adjustment = 0  # No pitch adjustment

    result = process_tts_and_rvc(text, selected_model, pitch_adjustment)
    print(f"Processed result: {result}")
