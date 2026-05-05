import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


# ───────────────────────────────────────────
# QUANTIZATION
# ───────────────────────────────────────────
def quantize(data, levels):
    min_val = np.min(data)
    max_val = np.max(data)

    step = (max_val - min_val) / levels

    # Guard: if step is 0 (all values identical, e.g. pure silence),
    # there is nothing to quantize — return the data unchanged.
    if step == 0:
        return data.copy()

    quantized = (
        np.round((data - min_val) / step) * step
        + min_val
    )

    # Replace any NaN/Inf that may have slipped through with 0
    quantized = np.nan_to_num(quantized, nan=0.0, posinf=0.0, neginf=0.0)

    return quantized


# ───────────────────────────────────────────
# RUN LENGTH ENCODING
# ───────────────────────────────────────────
def run_length_encode(data):
    flat = data.flatten()

    encoded = []
    count = 1

    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            encoded.append((flat[i - 1], count))
            count = 1

    encoded.append((flat[-1], count))

    return encoded


# ───────────────────────────────────────────
# AUDIO PIPELINE
# ───────────────────────────────────────────
def run_pipeline(params, log, progress, done):
    output_dir = params.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    frequency   = params.get("frequency", 440)
    noise_level = params.get("noise", 0.5)
    duration    = params.get("duration", 1.0)
    q_levels    = params.get("q_levels", 16)

    fs = 44100

    # ───────────────────────────────────────
    # STEP 1 → SIGNAL GENERATION / LOADING
    # ───────────────────────────────────────
    input_wav = params.get("input_wav", None)

    if input_wav and os.path.exists(input_wav):
        # ── Load user-provided WAV file ──────────────────────
        log(f"Loading audio file: {os.path.basename(input_wav)}")
        progress(10)

        file_fs, raw = wavfile.read(input_wav)

        # Normalise to float32 in the range [-1.0, 1.0]
        if raw.dtype == np.int16:
            raw = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            raw = raw.astype(np.float32) / 2147483648.0
        elif raw.dtype == np.uint8:
            raw = (raw.astype(np.float32) - 128.0) / 128.0
        else:
            raw = raw.astype(np.float32)

        # If stereo, mix down to mono by averaging the two channels
        if raw.ndim == 2:
            raw = raw.mean(axis=1)

        # Resample to 44100 Hz if the file uses a different sample rate
        if file_fs != fs:
            from scipy.signal import resample as scipy_resample
            num_samples = int(len(raw) * fs / file_fs)
            raw = scipy_resample(raw, num_samples)
            log(f"Resampled from {file_fs} Hz → {fs} Hz")

        # Trim to the user-requested duration (Duration spinbox).
        # If the file is shorter than requested, use the full file.
        max_samples = int(duration * fs)
        if len(raw) > max_samples:
            raw = raw[:max_samples]
            log(f"Trimmed to {duration:.1f} s ({max_samples} samples)")
        else:
            duration = len(raw) / fs
            log(f"Full file loaded: {duration:.2f} s")

        clean_with_silence = raw.copy()   # clean copy — before any noise is added
        
        # Add 0.5 seconds of silence to the end
        silence = np.zeros(fs // 2)
        clean_with_silence = np.concatenate((clean_with_silence, silence))
        time_axis          = np.linspace(0, duration + 0.5, len(clean_with_silence))

        # Optionally add noise on top of the loaded audio (controlled by Noise Level spinbox)
        if noise_level > 0:
            added_noise = np.random.normal(0, noise_level, raw.shape)
            noisy_signal = np.clip(raw + added_noise, -1.0, 1.0)
            noisy_signal = np.concatenate((noisy_signal, silence))
            log(f"Added noise (level={noise_level})")
        else:
            noisy_signal = np.concatenate((raw, silence))

        # Save a normalised copy of the (possibly noisy) input
        noisy_wav_path = os.path.join(output_dir, "input_audio.wav")
        wavfile.write(noisy_wav_path, fs, noisy_signal.astype(np.float32))
        log("Saved: input_audio.wav")

    else:
        # ── Generate a synthetic sine wave (original behaviour) ──
        log("Generating audio...")
        progress(10)

        t            = np.linspace(0, duration, int(fs * duration))
        clean_signal = np.sin(2 * np.pi * frequency * t)
        noise        = np.random.normal(0, noise_level, clean_signal.shape)
        silence      = np.zeros(fs // 2)

        noisy_signal       = np.concatenate((clean_signal + noise, silence))
        clean_with_silence = np.concatenate((clean_signal, silence))
        time_axis          = np.linspace(0, duration + 0.5, len(noisy_signal))

        # Save clean and noisy WAV files
        clean_wav_path = os.path.join(output_dir, "clean_signal.wav")
        noisy_wav_path = os.path.join(output_dir, "noisy_signal.wav")
        wavfile.write(clean_wav_path, fs, clean_with_silence.astype(np.float32))
        wavfile.write(noisy_wav_path, fs, noisy_signal.astype(np.float32))
        log("Saved: clean_signal.wav")
        log("Saved: noisy_signal.wav")

    # ───────────────────────────────────────
    # STEP 2 → STFT
    # ───────────────────────────────────────
    log("Applying STFT...")
    progress(30)

    f, t_stft, Zxx = signal.stft(noisy_signal, fs, nperseg=1024)
    magnitudes     = np.abs(Zxx)
    phases         = np.angle(Zxx)

    # ───────────────────────────────────────
    # STEP 3 → QUANTIZATION
    # ───────────────────────────────────────
    log("Quantizing frequency magnitudes...")
    progress(50)

    quantized = quantize(magnitudes, q_levels)

    # ───────────────────────────────────────
    # STEP 4 → RUN-LENGTH ENCODING
    # ───────────────────────────────────────
    log("Applying Run-Length Encoding...")
    progress(70)

    compressed = run_length_encode(quantized)

    # ───────────────────────────────────────
    # STEP 5 → RECONSTRUCTION
    # ───────────────────────────────────────
    log("Reconstructing audio...")
    progress(85)

    reconstructed_Zxx = quantized * np.exp(1j * phases)
    _, decoded_audio  = signal.istft(reconstructed_Zxx, fs)
    decoded_audio     = decoded_audio[:len(noisy_signal)]

    # Safety: replace any NaN/Inf produced by ISTFT, then clip to valid audio range
    decoded_audio = np.nan_to_num(decoded_audio, nan=0.0, posinf=0.0, neginf=0.0)
    decoded_audio = np.clip(decoded_audio, -1.0, 1.0)

    # Save compressed (reconstructed) WAV
    wav_path = os.path.join(output_dir, "compressed_audio.wav")
    wavfile.write(wav_path, fs, decoded_audio.astype(np.float32))
    log("Saved: compressed_audio.wav")

    # ───────────────────────────────────────
    # METRICS
    # ───────────────────────────────────────
    signal_power = np.sum(noisy_signal ** 2)
    noise_power  = np.sum((noisy_signal - decoded_audio) ** 2)

    if noise_power == 0:
        snr = "Infinite"
    else:
        snr = round(10 * np.log10(signal_power / noise_power), 2)

    original_size     = noisy_signal.size * 32
    compressed_size   = len(compressed) * 64
    compression_ratio = round(original_size / compressed_size, 2)

    # ───────────────────────────────────────
    # SAVE WAVEFORM PLOT
    # ───────────────────────────────────────
    log("Saving waveform plot...")
    fig1, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(time_axis, clean_with_silence, color="#2196F3")
    axs[0].set_title("Clean Signal")
    axs[0].set_ylabel("Amplitude")
    axs[1].plot(time_axis, noisy_signal, color="#F44336")
    axs[1].set_title("Noisy Signal")
    axs[1].set_ylabel("Amplitude")
    axs[2].plot(time_axis, decoded_audio, color="#4CAF50")
    axs[2].set_title("Decoded (Compressed) Signal")
    axs[2].set_ylabel("Amplitude")
    axs[2].set_xlabel("Time (s)")
    fig1.suptitle(f"Audio Waveforms — {frequency}Hz, noise={noise_level}, q_levels={q_levels}")
    fig1.tight_layout()
    waveform_path = os.path.join(output_dir, "waveforms.png")
    fig1.savefig(waveform_path, dpi=150)
    plt.close(fig1)
    log("Saved: waveforms.png")

    # ───────────────────────────────────────
    # SAVE SPECTROGRAM PLOT
    # ───────────────────────────────────────
    log("Saving spectrogram...")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(magnitudes, aspect="auto", origin="lower")
    axes[0].set_title("STFT Magnitudes (Original)")
    axes[0].set_xlabel("Time frame")
    axes[0].set_ylabel("Frequency bin")
    axes[1].imshow(quantized, aspect="auto", origin="lower")
    axes[1].set_title(f"Quantized Magnitudes ({q_levels} levels)")
    axes[1].set_xlabel("Time frame")
    axes[1].set_ylabel("Frequency bin")
    fig2.tight_layout()
    spectrogram_path = os.path.join(output_dir, "spectrogram.png")
    fig2.savefig(spectrogram_path, dpi=150)
    plt.close(fig2)
    log("Saved: spectrogram.png")

    progress(100)
    log("Audio compression completed.")
    log(f"All outputs saved to: {output_dir}/")

    # ───────────────────────────────────────
    # RETURN DATA TO GUI
    # ───────────────────────────────────────
    data = {
        "clean_signal":      clean_with_silence,
        "noisy_signal":      noisy_signal,
        "decoded_signal":    decoded_audio,
        "time_axis":         time_axis,
        "magnitudes":        magnitudes,
        "quantized":         quantized,
        "snr":               snr,
        "compression_ratio": compression_ratio,
        "rle_pairs":         len(compressed),
        "wav_path":          wav_path,
    }

    done(data)