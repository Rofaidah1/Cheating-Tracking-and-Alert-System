import os
import threading
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from detector import CheatingDetector

BASE_DIR = Path(__file__).resolve().parent
ALERTS_DIR = BASE_DIR / "alerts"
SCREENSHOTS_DIR = ALERTS_DIR / "screenshots"
OUTPUT_DIR = ALERTS_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

APP_BG = "#0b1220"
CARD_BG = "#111b2e"
CARD_2 = "#16233a"
TEXT = "#f4f7fb"
MUTED = "#8fa3bd"
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#38bdf8"
PURPLE = "#8b5cf6"
BORDER = "#243653"


class CheatingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CheatingGuard AI — Exam Monitoring")
        self.root.geometry("1180x760")
        self.root.minsize(1050, 680)
        self.root.configure(bg=APP_BG)

        self.video_path = None
        self.processing = False
        self.detector = None
        self.last_output = None
        self.alert_count = 0
        self.preview_window = None
        self.preview_cap = None
        self.preview_running = False
        self.preview_photo = None

        self.setup_style()
        self.build_ui()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Treeview",
            background=CARD_2,
            foreground=TEXT,
            fieldbackground=CARD_2,
            rowheight=32,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#1d2d49",
            foreground=TEXT,
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
        )
        style.map(
            "Treeview",
            background=[("selected", "#28466e")],
            foreground=[("selected", "white")],
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#1b2940",
            background=BLUE,
            bordercolor="#1b2940",
            lightcolor=BLUE,
            darkcolor=BLUE,
        )

    def make_button(self, parent, text, command, bg=BLUE, fg="#06111f",
                    width=None):
        b = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
            padx=16,
            pady=10,
        )
        if width:
            b.configure(width=width)
        return b

    def build_ui(self):

        header = tk.Frame(self.root, bg=APP_BG)
        header.pack(fill="x", padx=34, pady=(28, 10))

        left = tk.Frame(header, bg=APP_BG)
        left.pack(side="left")

        tk.Label(
            left,
            text="CHEATING",
            fg=BLUE,
            bg=APP_BG,
            font=("Segoe UI Black", 25),
        ).pack(side="left")
        tk.Label(
            left,
            text="GUARD",
            fg=TEXT,
            bg=APP_BG,
            font=("Segoe UI Black", 25),
        ).pack(side="left", padx=(5, 0))

        tk.Label(
            left,
            text="  AI EXAM MONITORING",
            fg=MUTED,
            bg=APP_BG,
            font=("Segoe UI Semibold", 10),
        ).pack(side="left", padx=14, pady=(12, 0))

        self.status_badge = tk.Label(
            header,
            text="●  SYSTEM READY",
            fg=GREEN,
            bg="#10251b",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=7,
        )
        self.status_badge.pack(side="right", pady=5)

        tk.Label(
            self.root,
            text="Detect • Track • Capture • Alert",
            fg=MUTED,
            bg=APP_BG,
            font=("Segoe UI", 12),
        ).pack(anchor="w", padx=36, pady=(0, 18))


        top = tk.Frame(self.root, bg=APP_BG)
        top.pack(fill="x", padx=34)

        self.card_video = self.card(top)
        self.card_video.pack(side="left", fill="both", expand=True, padx=(0, 9))

        self.card_stats = self.card(top)
        self.card_stats.pack(side="left", fill="both", expand=True, padx=(9, 0))


        tk.Label(
            self.card_video, text="EXAM VIDEO",
            fg=TEXT, bg=CARD_BG,
            font=("Segoe UI Semibold", 12)
        ).pack(anchor="w", padx=20, pady=(18, 4))

        self.video_label = tk.Label(
            self.card_video,
            text="No video selected\n\nUpload an exam recording to begin",
            fg=MUTED,
            bg="#0e1728",
            font=("Segoe UI", 12),
            height=6,
            justify="center",
        )
        self.video_label.pack(fill="x", padx=20, pady=10)

        actions = tk.Frame(self.card_video, bg=CARD_BG)
        actions.pack(fill="x", padx=20, pady=(2, 20))

        self.upload_btn = self.make_button(
            actions, "＋  Upload Video", self.upload_video, BLUE
        )
        self.upload_btn.pack(side="left")

        self.process_btn = self.make_button(
            actions, "▶  Start Analysis", self.process_video, GREEN
        )
        self.process_btn.pack(side="left", padx=8)

        self.history_btn = self.make_button(
            actions, "▣  Alert History", self.open_history, PURPLE, "white"
        )
        self.history_btn.pack(side="left")

        self.watch_btn = self.make_button(
            actions, "▶  Watch Caught Video", self.watch_processed_video, RED, "white"
        )
        self.watch_btn.pack(side="left", padx=8)
        self.watch_btn.configure(state="disabled")

        self.folder_btn = self.make_button(
            actions, "⌁  Screenshots", self.open_screenshots, "#263650", TEXT
        )
        self.folder_btn.pack(side="left", padx=8)


        tk.Label(
            self.card_stats, text="LIVE SYSTEM OVERVIEW",
            fg=TEXT, bg=CARD_BG,
            font=("Segoe UI Semibold", 12)
        ).pack(anchor="w", padx=20, pady=(18, 12))

        stats = tk.Frame(self.card_stats, bg=CARD_BG)
        stats.pack(fill="x", padx=20)

        self.alert_value = self.stat_box(stats, "ALERTS", "0", RED)
        self.alert_value.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.mode_value = self.stat_box(stats, "MODE", "READY", BLUE)
        self.mode_value.pack(side="left", fill="both", expand=True, padx=6)

        self.db_value = self.stat_box(stats, "DATABASE", "ONLINE", GREEN)
        self.db_value.pack(side="left", fill="both", expand=True, padx=(6, 0))

        tk.Label(
            self.card_stats,
            text="AI Pipeline",
            fg=MUTED, bg=CARD_BG,
            font=("Segoe UI Semibold", 10)
        ).pack(anchor="w", padx=20, pady=(20, 6))

        pipeline = tk.Frame(self.card_stats, bg="#0e1728")
        pipeline.pack(fill="x", padx=20, pady=(0, 20))

        for i, (name, color) in enumerate([
            ("YOLO Detection", BLUE),
            ("BoT-SORT Tracking", PURPLE),
            ("Cheating Logic", RED),
            ("SQLite Alerts", GREEN),
        ]):
            row = tk.Frame(pipeline, bg="#0e1728")
            row.pack(fill="x", padx=14, pady=7)
            tk.Label(row, text="●", fg=color, bg="#0e1728",
                     font=("Segoe UI", 11)).pack(side="left")
            tk.Label(row, text=name, fg=TEXT, bg="#0e1728",
                     font=("Segoe UI", 9)).pack(side="left", padx=8)
            tk.Label(row, text="READY", fg=GREEN, bg="#0e1728",
                     font=("Segoe UI Semibold", 8)).pack(side="right")


        process_card = self.card(self.root)
        process_card.pack(fill="x", padx=34, pady=18)

        header2 = tk.Frame(process_card, bg=CARD_BG)
        header2.pack(fill="x", padx=20, pady=(15, 5))

        tk.Label(
            header2, text="ANALYSIS PROGRESS",
            fg=TEXT, bg=CARD_BG,
            font=("Segoe UI Semibold", 11)
        ).pack(side="left")

        self.progress_text = tk.Label(
            header2, text="Waiting for video…",
            fg=MUTED, bg=CARD_BG,
            font=("Segoe UI", 9)
        )
        self.progress_text.pack(side="right")

        self.progress = ttk.Progressbar(
            process_card, orient="horizontal",
            mode="determinate", style="Horizontal.TProgressbar"
        )
        self.progress.pack(fill="x", padx=20, pady=(4, 16))


        console_card = self.card(self.root)
        console_card.pack(fill="both", expand=True, padx=34, pady=(0, 28))

        tk.Label(
            console_card, text="EVENT CONSOLE",
            fg=TEXT, bg=CARD_BG,
            font=("Segoe UI Semibold", 11)
        ).pack(anchor="w", padx=20, pady=(15, 8))

        self.console = tk.Text(
            console_card,
            bg="#09111f",
            fg="#b9c8da",
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            font=("Consolas", 9),
            height=6,
            padx=14,
            pady=10,
        )
        self.console.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        self.log("SYSTEM", "CheatingGuard AI initialized successfully.")
        self.log("MODEL", "YOLO + BoT-SORT pipeline ready.")
        self.log("DATABASE", "SQLite alert database ready.")

    def card(self, parent):
        return tk.Frame(
            parent,
            bg=CARD_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )

    def stat_box(self, parent, title, value, accent):
        f = tk.Frame(parent, bg="#0e1728",
                     highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            f, text=title, fg=MUTED, bg="#0e1728",
            font=("Segoe UI Semibold", 8)
        ).pack(anchor="w", padx=12, pady=(10, 1))
        value_label = tk.Label(
            f, text=value, fg=accent, bg="#0e1728",
            font=("Segoe UI Black", 16)
        )
        value_label.pack(anchor="w", padx=12, pady=(0, 10))
        f.value_label = value_label
        return f

    def log(self, tag, message):
        self.console.insert("end", f"[{tag:<8}] {message}\n")
        self.console.see("end")

    def set_status(self, text, color=GREEN, bg="#10251b"):
        self.status_badge.configure(text=f"●  {text}", fg=color, bg=bg)

    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select Exam Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.video_path = path
        name = Path(path).name
        self.video_label.configure(
            text=f"✓  {name}\n\nReady for AI analysis",
            fg=TEXT
        )
        self.log("VIDEO", f"Selected: {name}")
        self.progress["value"] = 0
        self.progress_text.configure(text="Video selected — ready to analyze")
        self.mode_value.value_label.configure(text="READY", fg=BLUE)

    def process_video(self):
        if self.processing:
            return

        if not self.video_path:
            messagebox.showwarning(
                "No Video Selected",
                "Please upload an exam video first."
            )
            return

        self.processing = True
        self.process_btn.configure(
            text="⏳  Processing…",
            state="disabled",
            bg="#36506e"
        )
        self.upload_btn.configure(state="disabled")
        self.set_status("AI ANALYSIS RUNNING", BLUE, "#102536")
        self.mode_value.value_label.configure(text="SCANNING", fg=BLUE)
        self.progress["value"] = 0
        self.log("AI", "Starting detection + tracking analysis…")

        stem = Path(self.video_path).stem
        output = OUTPUT_DIR / f"{stem}_processed.mp4"
        self.last_output = output

        def worker():
            try:
                self.detector = CheatingDetector()
                count = self.detector.process_video(
                    self.video_path,
                    output,
                    progress_callback=self.update_progress,
                )
                self.root.after(0, lambda: self.finish_processing(count, output))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self.processing_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def update_progress(self, value):
        self.root.after(
            0,
            lambda v=value: (
                self.progress.configure(value=max(0, min(100, v * 100))),
                self.progress_text.configure(
                    text=f"Analyzing exam video… {v*100:.0f}%"
                )
            )
        )

    def finish_processing(self, count, output):
        self.processing = False
        self.alert_count = count

        self.process_btn.configure(
            text="▶  Start Analysis",
            state="normal",
            bg=GREEN
        )
        self.upload_btn.configure(state="normal")
        self.watch_btn.configure(state="normal")
        self.alert_value.value_label.configure(
            text=str(count), fg=RED if count else GREEN
        )
        self.mode_value.value_label.configure(
            text="COMPLETE", fg=GREEN
        )
        self.progress["value"] = 100
        self.progress_text.configure(
            text=f"Analysis complete • {count} alert(s)"
        )
        self.set_status("ANALYSIS COMPLETE", GREEN, "#10251b")

        self.log(
            "RESULT",
            f"Analysis finished — {count} cheating alert(s) detected."
        )
        self.log("OUTPUT", f"Processed video: {output.name}")

        messagebox.showinfo(
            "Analysis Complete",
            f"AI analysis finished successfully.\n\n"
            f"Cheating alerts: {count}\n\n"
            f"Processed video saved to:\n{output}"
        )

    def processing_error(self, error):
        self.processing = False
        self.process_btn.configure(
            text="▶  Start Analysis",
            state="normal",
            bg=GREEN
        )
        self.upload_btn.configure(state="normal")
        self.watch_btn.configure(state="disabled")
        self.mode_value.value_label.configure(text="ERROR", fg=RED)
        self.set_status("PROCESSING ERROR", RED, "#32141a")
        self.log("ERROR", error)
        messagebox.showerror("Processing Error", error)

    def watch_processed_video(self):
        """Open an in-app player for the processed video with red/green/blue boxes."""
        if not self.last_output or not Path(self.last_output).exists():
            messagebox.showwarning(
                "No Processed Video",
                "Run Start Analysis first. The processed video will appear here."
            )
            return


        self.close_video_preview()

        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("CheatingGuard AI — Processed Exam Video")
        self.preview_window.geometry("1000x700")
        self.preview_window.configure(bg=APP_BG)
        self.preview_window.protocol("WM_DELETE_WINDOW", self.close_video_preview)

        top = tk.Frame(self.preview_window, bg=APP_BG)
        top.pack(fill="x", padx=18, pady=(14, 8))

        tk.Label(
            top,
            text="LIVE REVIEW  •  PROCESSED EXAM VIDEO",
            fg=TEXT,
            bg=APP_BG,
            font=("Segoe UI Semibold", 13),
        ).pack(side="left")

        self.preview_status = tk.Label(
            top,
            text="● PLAYING",
            fg=GREEN,
            bg="#10251b",
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
        )
        self.preview_status.pack(side="right")

        self.preview_label = tk.Label(
            self.preview_window,
            text="Loading processed video…",
            bg="#050a12",
            fg=MUTED,
            font=("Segoe UI", 12),
        )
        self.preview_label.pack(fill="both", expand=True, padx=18, pady=8)

        controls = tk.Frame(self.preview_window, bg=APP_BG)
        controls.pack(fill="x", padx=18, pady=(4, 15))

        self.preview_play_btn = self.make_button(
            controls, "❚❚  Pause", self.toggle_preview, BLUE
        )
        self.preview_play_btn.pack(side="left")

        self.preview_restart_btn = self.make_button(
            controls, "↻  Restart", self.restart_preview, PURPLE, "white"
        )
        self.preview_restart_btn.pack(side="left", padx=8)

        tk.Label(
            controls,
            text="🔵 Phone   🔴 Cheating Student   🟢 Normal Student",
            fg=MUTED,
            bg=APP_BG,
            font=("Segoe UI", 9),
        ).pack(side="right")

        self.preview_cap = cv2.VideoCapture(str(self.last_output))
        if not self.preview_cap.isOpened():
            self.close_video_preview()
            messagebox.showerror(
                "Video Error",
                "Could not open the processed video."
            )
            return

        self.preview_running = True
        self.show_next_video_frame()

    def show_next_video_frame(self):
        if not self.preview_window or not self.preview_cap:
            return

        if not self.preview_running:
            return

        ok, frame = self.preview_cap.read()
        if not ok:
            self.preview_status.configure(
                text="● FINISHED", fg=BLUE, bg="#102536"
            )
            self.preview_play_btn.configure(text="▶  Play")
            self.preview_running = False
            return


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        max_w = max(600, self.preview_window.winfo_width() - 40)
        max_h = max(450, self.preview_window.winfo_height() - 150)
        h, w = frame_rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        image = Image.fromarray(frame_rgb).resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )
        self.preview_photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.preview_photo, text="")

        fps = self.preview_cap.get(cv2.CAP_PROP_FPS) or 25
        delay_ms = max(10, int(1000 / fps))
        self.preview_window.after(delay_ms, self.show_next_video_frame)

    def toggle_preview(self):
        if not self.preview_cap:
            return

        self.preview_running = not self.preview_running

        if self.preview_running:
            self.preview_play_btn.configure(text="❚❚  Pause")
            self.preview_status.configure(
                text="● PLAYING", fg=GREEN, bg="#10251b"
            )
            self.show_next_video_frame()
        else:
            self.preview_play_btn.configure(text="▶  Play")
            self.preview_status.configure(
                text="● PAUSED", fg=BLUE, bg="#102536"
            )

    def restart_preview(self):
        if not self.preview_cap:
            return

        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.preview_running = True
        self.preview_play_btn.configure(text="❚❚  Pause")
        self.preview_status.configure(
            text="● PLAYING", fg=GREEN, bg="#10251b"
        )
        self.show_next_video_frame()

    def close_video_preview(self):
        self.preview_running = False

        if self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None

        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except Exception:
                pass
            self.preview_window = None

        self.preview_photo = None

    def open_screenshots(self):
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(SCREENSHOTS_DIR))
        except Exception:
            messagebox.showinfo(
                "Screenshots Folder",
                str(SCREENSHOTS_DIR)
            )

    def open_history(self):

        history = project_path("alert_history.py")
        if history.exists():
            subprocess.Popen(["python", str(history)], cwd=str(BASE_DIR))
        else:
            messagebox.showinfo(
                "Alert History",
                "Open the Alert History module from the project."
            )


def project_path(name):
    return BASE_DIR / name


if __name__ == "__main__":
    root = tk.Tk()
    app = CheatingApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close_video_preview(), root.destroy()))
    root.mainloop()
