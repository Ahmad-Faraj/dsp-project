import os
import threading

import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Audio_Compression.audio_encoder import run_pipeline as run_audio
from Video_Compression.video_encoder import run_pipeline as run_video


# ───────────────────────────────────────────
# THREADING
# ───────────────────────────────────────────
def run_async(func, params, log, progress, done, root):
    threading.Thread(
        target=func,
        args=(
            params,
            lambda m: root.after(0, log, m),
            lambda v: root.after(0, progress, v),
            lambda *d: root.after(0, done, *d),
        ),
        daemon=True,
    ).start()


# ───────────────────────────────────────────
# MATPLOTLIB DISPLAY
# ───────────────────────────────────────────
def draw_figure(container, fig):
    for widget in container.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ───────────────────────────────────────────
# AUDIO TAB
# ───────────────────────────────────────────
class AudioTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.build_ui()

    def build_ui(self):
        left  = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Notebook(self)
        right.pack(side="right", fill="both", expand=True)

        self.wave_tab  = ttk.Frame(right)
        self.spec_tab  = ttk.Frame(right)
        self.stats_tab = ttk.Frame(right)

        right.add(self.wave_tab,  text="Waveforms")
        right.add(self.spec_tab,  text="Spectrogram")
        right.add(self.stats_tab, text="Statistics")

        self.frequency  = tk.DoubleVar(value=440)
        self.noise      = tk.DoubleVar(value=0.5)
        self.duration   = tk.DoubleVar(value=1.0)
        self.bits       = tk.IntVar(value=16)
        self.input_wav  = tk.StringVar()

        # ── Audio file picker ────────────────────────────────
        ttk.Label(left, text="Input Audio File (optional)").pack(anchor="w", pady=(6, 0))
        file_row = ttk.Frame(left)
        file_row.pack(fill="x")
        self.file_label = ttk.Label(file_row, text="No file selected", foreground="gray",
                                    wraplength=200, justify="left")
        self.file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Browse", command=self.pick_audio).pack(side="right")
        ttk.Button(left, text="Clear File (use generated signal)",
                   command=self.clear_audio).pack(fill="x", pady=(2, 8))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="Generated Signal Settings").pack(anchor="w")

        # ── Generated signal controls ────────────────────────
        ttk.Label(left, text="Frequency").pack(anchor="w")
        self.freq_spin = ttk.Spinbox(left, from_=100, to=5000, textvariable=self.frequency)
        self.freq_spin.pack(fill="x")

        ttk.Label(left, text="Noise Level").pack(anchor="w")
        self.noise_spin = ttk.Spinbox(left, from_=0, to=2, increment=0.1, textvariable=self.noise)
        self.noise_spin.pack(fill="x")

        ttk.Label(left, text="Duration").pack(anchor="w")
        self.dur_spin = ttk.Spinbox(left, from_=0.5, to=5, increment=0.5, textvariable=self.duration)
        self.dur_spin.pack(fill="x")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)

        ttk.Label(left, text="Quantization Levels").pack(anchor="w")
        ttk.Combobox(left, values=[4, 8, 16, 32, 64], textvariable=self.bits, state="readonly").pack(fill="x")

        ttk.Button(left, text="Run Audio Compression", command=self.run).pack(fill="x", pady=10)

        self.progress = ttk.Progressbar(left, maximum=100)
        self.progress.pack(fill="x")

        self.log = tk.Text(left, height=10, width=35)
        self.log.pack(fill="both", expand=True, pady=10)

    def log_message(self, message):
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def update_progress(self, value):
        self.progress["value"] = value

    def pick_audio(self):
        path = filedialog.askopenfilename(
            title="Select a WAV audio file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.input_wav.set(path)
            self.file_label.config(text=os.path.basename(path), foreground="black")
            self.log_message(f"File selected: {os.path.basename(path)}")
            self.log_message("Tip: Noise adds static on top of your file. Duration trims it.")

    def clear_audio(self):
        self.input_wav.set("")
        self.file_label.config(text="No file selected", foreground="gray")
        self.log_message("Cleared — will use generated signal.")

    def run(self):
        params = {
            "frequency":  self.frequency.get(),
            "noise":      self.noise.get(),
            "duration":   self.duration.get(),
            "q_levels":   self.bits.get(),
            "output_dir": "outputs",
        }

        # Only pass input_wav if the user actually selected a file
        wav = self.input_wav.get()
        if wav and os.path.exists(wav):
            params["input_wav"] = wav

        run_async(
            run_audio, params,
            self.log_message, self.update_progress, self.finished,
            self.winfo_toplevel(),
        )

    def finished(self, data):
        fig1, axs = plt.subplots(3, 1, figsize=(8, 6))
        axs[0].plot(data["time_axis"], data["clean_signal"])
        axs[0].set_title("Clean Signal")
        axs[1].plot(data["time_axis"], data["noisy_signal"])
        axs[1].set_title("Noisy Signal")
        axs[2].plot(data["time_axis"], data["decoded_signal"])
        axs[2].set_title("Decoded Signal")
        fig1.tight_layout()
        draw_figure(self.wave_tab, fig1)

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(data["magnitudes"], aspect="auto", origin="lower")
        ax.set_title("STFT Magnitudes")
        draw_figure(self.spec_tab, fig2)

        for widget in self.stats_tab.winfo_children():
            widget.destroy()

        for s in [
            f"SNR: {data['snr']} dB",
            f"Compression Ratio: {data['compression_ratio']} : 1",
            f"RLE Pairs: {data['rle_pairs']}",
        ]:
            ttk.Label(self.stats_tab, text=s, font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)

        ttk.Separator(self.stats_tab, orient="horizontal").pack(fill="x", padx=10, pady=10)

        wav_path   = data["wav_path"]
        output_dir = os.path.dirname(os.path.abspath(wav_path))

        ttk.Label(
            self.stats_tab,
            text=f"Output folder:  {output_dir}",
            font=("Arial", 10), foreground="gray"
        ).pack(anchor="w", padx=10)

        btn_frame = ttk.Frame(self.stats_tab)
        btn_frame.pack(anchor="w", padx=10, pady=8)

        ttk.Button(
            btn_frame, text="📂  Open Output Folder",
            command=lambda: os.startfile(output_dir)
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_frame, text="▶  Play Compressed Audio",
            command=lambda: os.startfile(wav_path)
        ).pack(side="left")


# ───────────────────────────────────────────
# VIDEO TAB
# ───────────────────────────────────────────
class VideoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.build_ui()

    def build_ui(self):
        left  = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Notebook(self)
        right.pack(side="right", fill="both", expand=True)

        self.psnr_tab    = ttk.Frame(right)
        self.frame_tab   = ttk.Frame(right)
        self.compare_tab = ttk.Frame(right)
        self.stats_tab   = ttk.Frame(right)

        right.add(self.psnr_tab,    text="PSNR")
        right.add(self.frame_tab,   text="Frame Types")
        right.add(self.compare_tab, text="Frame Comparison")
        right.add(self.stats_tab,   text="Statistics")

        self.video_path  = tk.StringVar()
        self.max_frames  = tk.IntVar(value=30)
        self.i_interval  = tk.IntVar(value=10)
        self.block_size  = tk.IntVar(value=16)
        self.search_area = tk.IntVar(value=8)

        ttk.Button(left, text="Select Video", command=self.pick_video).pack(fill="x")

        ttk.Label(left, text="Max Frames").pack(anchor="w")
        ttk.Spinbox(left, from_=1, to=500, textvariable=self.max_frames).pack(fill="x")

        ttk.Label(left, text="I-Frame Interval").pack(anchor="w")
        ttk.Spinbox(left, from_=1, to=60, textvariable=self.i_interval).pack(fill="x")

        ttk.Label(left, text="Block Size").pack(anchor="w")
        ttk.Combobox(left, values=[8, 16, 32], textvariable=self.block_size, state="readonly").pack(fill="x")

        ttk.Label(left, text="Search Area").pack(anchor="w")
        ttk.Spinbox(left, from_=2, to=32, textvariable=self.search_area).pack(fill="x")

        ttk.Button(left, text="Run Video Compression", command=self.run).pack(fill="x", pady=10)

        self.progress = ttk.Progressbar(left, maximum=100)
        self.progress.pack(fill="x")

        self.log = tk.Text(left, height=12, width=35)
        self.log.pack(fill="both", expand=True, pady=10)

    def pick_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if path:
            self.video_path.set(path)
            self.log_message(f"Loaded: {os.path.basename(path)}")

    def log_message(self, message):
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def update_progress(self, value):
        self.progress["value"] = value

    def run(self):
        if not os.path.exists(self.video_path.get()):
            self.log_message("Please select a valid video.")
            return

        params = {
            "video_path":  self.video_path.get(),
            "max_frames":  self.max_frames.get(),
            "i_interval":  self.i_interval.get(),
            "block_size":  self.block_size.get(),
            "search_area": self.search_area.get(),
            "output_dir":  "outputs",
        }

        run_async(
            run_video, params,
            self.log_message, self.update_progress, self.finished,
            self.winfo_toplevel(),
        )

    def finished(self, data, summary):
        # ── PSNR chart ──────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(data["psnr_values"])
        ax1.set_title("PSNR Per Frame")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("PSNR (dB)")
        draw_figure(self.psnr_tab, fig1)

        # ── Frame type chart ────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        frame_values = [1 if f == "I" else 0 for f in data["frame_types"]]
        ax2.step(range(len(frame_values)), frame_values, where="mid")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["P", "I"])
        ax2.set_title("Frame Types")
        draw_figure(self.frame_tab, fig2)

        # ── Frame comparison (original vs reconstructed) ──
        pairs = data["frame_pairs"]
        n     = len(pairs)
        if n > 0:
            fig3, axes = plt.subplots(2, n, figsize=(3 * n, 6))
            if n == 1:
                axes = np.expand_dims(axes, axis=1)
            for col, (orig, recon) in enumerate(pairs):
                axes[0, col].imshow(orig,  cmap="gray", vmin=0, vmax=255)
                axes[0, col].set_title(f"Original #{col}", fontsize=8)
                axes[0, col].axis("off")
                axes[1, col].imshow(recon, cmap="gray", vmin=0, vmax=255)
                axes[1, col].set_title(f"Recon #{col}",   fontsize=8)
                axes[1, col].axis("off")
            fig3.suptitle("Original vs Reconstructed Frames (Y channel)")
            fig3.tight_layout()
            draw_figure(self.compare_tab, fig3)

        # ── Statistics ──────────────────────────────
        for widget in self.stats_tab.winfo_children():
            widget.destroy()

        for s in [
            f"Average PSNR: {summary['avg_psnr']:.2f} dB",
            f"Compression Ratio: {summary['ratio']} : 1",
            f"Total Frames: {summary['total_frames']}",
            f"Bitstream: {data['bitstream_path']}",
        ]:
            ttk.Label(self.stats_tab, text=s, font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)


# ───────────────────────────────────────────
# MAIN APP
# ───────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Multimedia Compression Studio")
        self.geometry("1200x700")

        style = ttk.Style()
        style.theme_use("clam")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        notebook.add(AudioTab(notebook), text="Audio")
        notebook.add(VideoTab(notebook), text="Video")


if __name__ == "__main__":
    app = App()
    app.mainloop()