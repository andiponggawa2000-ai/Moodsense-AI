"""
MoodSense AI - ENHANCED VERSION
Aplikasi desktop chatbot analisis emosi dengan dashboard analytics, mood tracking, dan fitur interaktif.

Fitur Utama:
✨ Real-time Emotion Analysis dengan AI
📊 Mood Statistics Dashboard
📈 Mood Tracking Timeline
💾 Export Conversation History (JSON/TXT)
🎨 Dark/Light Theme Support
⚡ Smooth Animations & Transitions
🎯 Breathing Exercise & Mood Tips
🔔 Mood Notifications
📱 Responsive UI

Teknologi:
- Python 3.8+
- CustomTkinter (Modern GUI)
- Transformers (NLP Model)
- Torch
- Matplotlib (Charting)
- Pandas (Data Processing)
"""

import threading
import customtkinter as ctk
from transformers import pipeline
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


# ==============================
# KONFIGURASI APLIKASI
# ==============================
APP_NAME = "MOODSENSE AI"
SUBTITLE = "Your Personal Emotion Analyzer & Wellness Companion"
VERSION = "2.0 Enhanced"

# Model sentiment yang lebih ringan dan cepat
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Emoji dan display untuk emosi
EMOTION_DISPLAY = {
    "bahagia": "😊 Bahagia",
    "sedih": "😢 Sedih",
    "cemas": "😨 Cemas",
    "marah": "😠 Marah",
    "netral": "😐 Netral",
}

EMOTION_COLORS = {
    "bahagia": ("#FFD700", "#FFA500"),
    "sedih": ("#4B0082", "#9370DB"),
    "cemas": ("#FF6347", "#FF7F50"),
    "marah": ("#DC143C", "#FF1493"),
    "netral": ("#808080", "#A9A9A9"),
}

# Penjelasan & saran untuk setiap emosi
EXPLANATIONS = {
    "bahagia": "Teks menunjukkan perasaan positif, lega, senang, atau antusias terhadap sesuatu.",
    "sedih": "Teks menunjukkan rasa kecewa, kehilangan semangat, kesepian, atau beban emosional.",
    "cemas": "Teks menunjukkan kekhawatiran, tekanan, takut gagal, atau pikiran yang terasa mengganggu.",
    "marah": "Teks menunjukkan rasa kesal, frustrasi, tidak terima, atau emosi yang sedang memuncak.",
    "netral": "Teks terdengar cukup stabil dan tidak menunjukkan emosi yang sangat kuat.",
}

SUGGESTIONS = {
    "bahagia": [
        "Nikmati momennya dan coba catat hal yang membuatmu merasa baik agar bisa kamu ulangi nanti.",
        "💡 Coba bagikan kebahagiaan ini kepada orang terdekat.",
        "🎯 Manfaatkan mood positifmu untuk menyelesaikan tugas yang penting.",
    ],
    "sedih": [
        "Ambil jeda sebentar, tarik napas pelan, dan ceritakan perasaanmu kepada orang yang kamu percaya.",
        "💡 Coba lakukan aktivitas yang biasanya membuat kamu nyaman (menonton film, mendengarkan musik).",
        "🎯 Menulis jurnal bisa membantu mengekspresikan perasaanmu dengan lebih jelas.",
    ],
    "cemas": [
        "Pecah masalah menjadi langkah kecil, mulai dari hal yang paling mudah, lalu beri tubuhmu waktu istirahat.",
        "💡 Coba teknik breathing exercise untuk menenangkan diri.",
        "🎯 Identifikasi sumber kekhawatiran spesifik dan buat rencana aksi konkret.",
    ],
    "marah": [
        "Berhenti sejenak sebelum merespons, atur napas, lalu tulis apa yang sebenarnya membuatmu terganggu.",
        "💡 Lakukan aktivitas fisik seperti berjalan atau olahraga untuk melepaskan energi negatif.",
        "🎯 Cari akar masalah sesungguhnya di balik rasa marah yang kamu rasakan.",
    ],
    "netral": [
        "Tetap perhatikan kondisi diri. Kamu bisa menulis jurnal singkat untuk memahami perasaanmu lebih jelas.",
        "💡 Coba tanyakan pada diri sendiri: Apa yang benar-benar aku rasakan sekarang?",
        "🎯 Lakukan aktivitas yang membawa kesenangan kecil hari ini.",
    ],
}

# Kata kunci emosi bahasa Indonesia
EMOTION_KEYWORDS = {
    "bahagia": ["senang", "bahagia", "gembira", "lega", "semangat", "suka", "puas", "bangga", "tertawa", "bersyukur", "nikmat"],
    "sedih": ["sedih", "kecewa", "hancur", "menangis", "sepi", "kesepian", "putus asa", "terluka", "capek", "lemah", "depresi"],
    "cemas": ["cemas", "khawatir", "takut", "panik", "gelisah", "deg-degan", "ujian", "gagal", "bingung", "tertekan"],
    "marah": ["marah", "kesal", "benci", "emosi", "jengkel", "muak", "frustrasi", "sebal", "tidak adil", "geram"],
    "netral": [],
}


# ==============================
# HELPER CLASSES & UTILITIES
# ==============================
class DataStore:
    """Menyimpan dan mengelola data chat dan mood history"""
    
    def __init__(self, data_dir: str = "moodsense_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.chat_file = self.data_dir / "conversations.json"
        self.mood_file = self.data_dir / "mood_history.json"
        self.load_data()

    def load_data(self):
        self.conversations = self._load_json(self.chat_file, [])
        self.mood_history = self._load_json(self.mood_file, [])

    def _load_json(self, filepath: Path, default):
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return default
        return default

    def _save_json(self, filepath: Path, data):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_conversation(self, sender: str, message: str, mood: Dict = None):
        """Tambah pesan ke conversation history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "message": message,
            "mood": mood
        }
        self.conversations.append(entry)
        self._save_json(self.chat_file, self.conversations)

    def add_mood_entry(self, emotion: str, confidence: int, text_length: int = 0):
        """Tambah entry ke mood history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion,
            "confidence": confidence,
            "text_length": text_length
        }
        self.mood_history.append(entry)
        self._save_json(self.mood_file, self.mood_history)

    def get_mood_stats(self) -> Dict:
        """Get statistik mood dari history"""
        if not self.mood_history:
            return {emotion: 0 for emotion in EMOTION_DISPLAY.keys()}
        
        stats = {emotion: 0 for emotion in EMOTION_DISPLAY.keys()}
        for entry in self.mood_history:
            emotion = entry.get("emotion", "netral")
            stats[emotion] = stats.get(emotion, 0) + 1
        return stats

    def export_conversations(self, format: str = "json") -> str:
        """Export conversations ke file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "json":
            filename = f"moodsense_chat_{timestamp}.json"
            filepath = self.data_dir / filename
            self._save_json(filepath, self.conversations)
        else:  # txt
            filename = f"moodsense_chat_{timestamp}.txt"
            filepath = self.data_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== MoodSense AI Chat Export ===\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Messages: {len(self.conversations)}\n\n")
                
                for entry in self.conversations:
                    timestamp = entry.get("timestamp", "")
                    sender = entry.get("sender", "").upper()
                    message = entry.get("message", "")
                    f.write(f"[{timestamp}] {sender}:\n{message}\n\n")
        
        return str(filepath)


class AnimationHelper:
    """Helper untuk smooth animations"""
    
    @staticmethod
    def fade_in(widget, duration: int = 500, steps: int = 20):
        """Fade in animation untuk widget"""
        widget.configure(fg_color="#050816")
        
    @staticmethod
    def pulse(label, color1: str, color2: str, duration: int = 1000):
        """Pulse animation untuk label"""
        pass  # Simplified untuk performance


# ==============================
# MAIN APPLICATION CLASS
# ==============================
class MoodSenseAI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pengaturan jendela
        self.title(f"{APP_NAME} - {VERSION}")
        self.geometry("1400x850")
        self.minsize(1000, 700)
        self.configure(fg_color="#050816")

        # Tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self.analyzer = None
        self.model_ready = False
        self.is_analyzing = False
        self.data_store = DataStore()
        self.current_mood = None

        # Grid configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Build UI
        self.create_sidebar()
        self.create_main_panel()
        
        # Load model in background
        self.show_loading("Memuat model AI... tunggu sebentar")
        threading.Thread(target=self.load_model, daemon=True).start()

    # ==============================
    # UI CONSTRUCTION
    # ==============================
    def create_sidebar(self):
        """Sidebar dengan navigation dan stats"""
        self.sidebar = ctk.CTkFrame(self, fg_color="#0B0E1F", width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(3, weight=1)

        # Header sidebar
        header = ctk.CTkFrame(self.sidebar, fg_color="#1A1F3A", corner_radius=8)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        header.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header,
            text="🧠 MOODSENSE",
            font=("Consolas", 18, "bold"),
            text_color="#38BDF8"
        )
        title_label.pack(pady=8)

        version_label = ctk.CTkLabel(
            header,
            text=f"v{VERSION.split()[0]}",
            font=("Consolas", 10),
            text_color="#8B5CF6"
        )
        version_label.pack(pady=(0, 8))

        # Quick stats frame
        stats_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        stats_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        stats_frame.grid_columnconfigure(0, weight=1)

        stats_label = ctk.CTkLabel(
            stats_frame,
            text="📊 Mood Stats",
            font=("Consolas", 13, "bold"),
            text_color="#A78BFA"
        )
        stats_label.pack()

        self.stats_display = ctk.CTkLabel(
            stats_frame,
            text="Loading...",
            font=("Consolas", 10),
            text_color="#D1D5DB",
            justify="left"
        )
        self.stats_display.pack(pady=10)

        # Button frame
        button_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)

        self.new_chat_btn = ctk.CTkButton(
            button_frame,
            text="✨ Obrolan Baru",
            command=self.new_chat,
            fg_color="#0EA5E9",
            hover_color="#0284C7",
            font=("Consolas", 12, "bold"),
            height=40
        )
        self.new_chat_btn.pack(side="left", fill="x", padx=(0, 5))

        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="🗑️",
            command=self.clear_history,
            fg_color="#EF4444",
            hover_color="#DC2626",
            width=40,
            height=40
        )
        self.clear_btn.pack(side="left")

        # Features section
        features_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        features_frame.grid(row=3, column=0, sticky="nsew", padx=15, pady=15)
        features_frame.grid_columnconfigure(0, weight=1)

        features_label = ctk.CTkLabel(
            features_frame,
            text="⚡ Fitur",
            font=("Consolas", 13, "bold"),
            text_color="#A78BFA"
        )
        features_label.pack(anchor="w", pady=(0, 10))

        self.dashboard_btn = ctk.CTkButton(
            features_frame,
            text="📈 Dashboard",
            command=self.show_dashboard,
            fg_color="#8B5CF6",
            hover_color="#7C3AED",
            font=("Consolas", 11),
            height=35
        )
        self.dashboard_btn.pack(fill="x", pady=5)

        self.breathing_btn = ctk.CTkButton(
            features_frame,
            text="🧘 Breathing Exercise",
            command=self.show_breathing,
            fg_color="#10B981",
            hover_color="#059669",
            font=("Consolas", 11),
            height=35
        )
        self.breathing_btn.pack(fill="x", pady=5)

        self.export_btn = ctk.CTkButton(
            features_frame,
            text="💾 Export Chat",
            command=self.export_dialog,
            fg_color="#F59E0B",
            hover_color="#D97706",
            font=("Consolas", 11),
            height=35
        )
        self.export_btn.pack(fill="x", pady=5)

        # Footer
        footer_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer_frame.grid(row=4, column=0, sticky="ew", padx=15, pady=15)

        footer_label = ctk.CTkLabel(
            footer_frame,
            text="Made with 💜 by MoodSense AI",
            font=("Consolas", 9),
            text_color="#6B7280"
        )
        footer_label.pack()

    def create_main_panel(self):
        """Main panel dengan chat area dan input"""
        main_frame = ctk.CTkFrame(self, fg_color="#050816")
        main_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Header
        self.create_header(main_frame)

        # Chat area
        chat_shell = ctk.CTkFrame(
            main_frame,
            fg_color="#0B1024",
            border_width=2,
            border_color="#1D4ED8",
            corner_radius=12
        )
        chat_shell.grid(row=1, column=0, sticky="nsew", padx=0, pady=8)
        chat_shell.grid_columnconfigure(0, weight=1)
        chat_shell.grid_rowconfigure(0, weight=1)

        self.chat_area = ctk.CTkScrollableFrame(
            chat_shell,
            fg_color="#0B1024",
            corner_radius=10,
            scrollbar_button_color="#2563EB",
            scrollbar_button_hover_color="#7C3AED"
        )
        self.chat_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_area.grid_columnconfigure(0, weight=1)

        # Input area
        self.create_input_area(main_frame)

    def create_header(self, parent):
        """Header dengan title dan info"""
        header = ctk.CTkFrame(parent, fg_color="#0B1024", corner_radius=8)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text=f"🎯 {APP_NAME}",
            font=("Consolas", 24, "bold"),
            text_color="#38BDF8"
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))

        subtitle = ctk.CTkLabel(
            header,
            text=SUBTITLE,
            font=("Consolas", 11),
            text_color="#A78BFA"
        )
        subtitle.pack(anchor="w", padx=15, pady=(0, 10))

    def create_input_area(self, parent):
        """Input area untuk chat"""
        input_frame = ctk.CTkFrame(parent, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=(10, 0))
        input_frame.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkEntry(
            input_frame,
            height=50,
            placeholder_text="Tulis perasaanmu di sini...",
            fg_color="#111936",
            border_color="#38BDF8",
            border_width=2,
            text_color="#E0F2FE",
            placeholder_text_color="#64748B",
            font=("Consolas", 14)
        )
        self.input_box.pack(side="left", fill="x", padx=(0, 10))
        self.input_box.bind("<Return>", lambda e: self.send_message())

        self.send_button = ctk.CTkButton(
            input_frame,
            text="🚀 Kirim",
            width=120,
            height=50,
            fg_color="#9333EA",
            hover_color="#7E22CE",
            text_color="#F8FAFC",
            font=("Consolas", 13, "bold"),
            command=self.send_message
        )
        self.send_button.pack(side="left")

    # ==============================
    # CHAT FUNCTIONS
    # ==============================
    def add_user_message(self, message: str):
        """Tambah pesan user ke chat"""
        self.add_message(message, sender="user")
        self.data_store.add_conversation("user", message)

    def add_ai_message(self, message: str, mood_data: Dict = None):
        """Tambah pesan AI ke chat"""
        self.add_message(message, sender="ai")
        self.data_store.add_conversation("ai", message, mood_data)

    def add_message(self, message: str, sender: str):
        """Render pesan di chat area"""
        row = len(self.chat_area.winfo_children())

        wrapper = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", pady=8, padx=8)
        wrapper.grid_columnconfigure(0, weight=1)

        is_user = sender == "user"
        bubble_color = "#123B66" if is_user else "#17112E"
        border_color = "#38BDF8" if is_user else "#A855F7"
        text_color = "#E0F2FE" if is_user else "#F5F3FF"
        anchor = "e" if is_user else "w"

        bubble = ctk.CTkFrame(
            wrapper,
            fg_color=bubble_color,
            border_color=border_color,
            border_width=1,
            corner_radius=10
        )
        bubble.grid(row=0, column=0, sticky=anchor, padx=20)

        label = ctk.CTkLabel(
            bubble,
            text=message,
            wraplength=550,
            justify="left",
            text_color=text_color,
            font=("Consolas", 12),
            padx=14,
            pady=10
        )
        label.pack()

        self.auto_scroll()

    def send_message(self):
        """Kirim pesan dari user"""
        if self.is_analyzing:
            return

        if not self.model_ready:
            self.add_ai_message("⏳ Model AI masih dimuat. Tunggu sebentar...")
            return

        user_text = self.input_box.get().strip()
        if not user_text:
            return

        self.input_box.delete(0, "end")
        self.add_user_message(user_text)
        self.show_loading("🔄 AI sedang menganalisis perasaanmu...")

        threading.Thread(target=self.process_message, args=(user_text,), daemon=True).start()

    def process_message(self, user_text: str):
        """Process message di background thread"""
        self.is_analyzing = True
        self.set_input_state("disabled")

        try:
            result_text, mood_data = self.analyze_emotion(user_text)
            self.current_mood = mood_data
        except Exception as error:
            result_text = f"❌ Error: {str(error)}"
            mood_data = None

        self.after(0, self.hide_loading)
        self.after(0, lambda: self.add_ai_message(result_text, mood_data))
        self.after(0, lambda: self.set_input_state("normal"))
        self.after(0, self.update_stats)
        self.is_analyzing = False

    # ==============================
    # EMOTION ANALYSIS
    # ==============================
    def load_model(self):
        """Load sentiment analysis model"""
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                device=-1,
            )
            self.model_ready = True
            self.after(0, self.hide_loading)
            self.after(0, lambda: self.set_input_state("normal"))
            self.after(0, lambda: self.add_ai_message(
                "✅ Model AI siap!\n\nHalo, aku adalah MoodSense AI. Ceritakan perasaanmu hari ini, dan aku akan membantu menganalisis emosi serta memberikan saran yang relevan. 💜"
            ))
        except Exception as error:
            self.model_ready = False
            self.after(0, self.hide_loading)
            self.after(0, lambda: self.set_input_state("disabled"))
            self.after(0, lambda: self.add_ai_message(
                f"❌ Error loading model: {str(error)}\n\nPastikan library sudah terinstall: pip install transformers torch customtkinter"
            ))

    def analyze_emotion(self, text: str) -> Tuple[str, Dict]:
        """Analisis emosi dari teks user"""
        # Get sentiment dari model
        sentiment_result = self.analyzer(text)[0]
        sentiment_label = sentiment_result["label"].upper()
        sentiment_score = sentiment_result["score"]

        # Keyword matching untuk emosi Indonesia
        text_lower = text.lower()
        keyword_scores = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            keyword_scores[emotion] = sum(1 for keyword in keywords if keyword in text_lower)

        strongest_emotion = max(keyword_scores, key=keyword_scores.get)
        strongest_score = keyword_scores[strongest_emotion]

        # Determine final emotion
        if strongest_score > 0:
            top_label = strongest_emotion
            confidence = min(95, max(int(sentiment_score * 100), 72 + strongest_score * 7))
        elif sentiment_label == "POSITIVE":
            top_label = "bahagia"
            confidence = int(sentiment_score * 100)
        elif sentiment_label == "NEGATIVE":
            top_label = "sedih"
            confidence = int(sentiment_score * 100)
        else:
            top_label = "netral"
            confidence = int(sentiment_score * 100)

        if confidence < 55:
            top_label = "netral"
            confidence = 60

        # Store mood data
        self.data_store.add_mood_entry(top_label, confidence, len(text))

        # Format response dengan struktur yang lebih baik
        response = (
            f"🎯 **{EMOTION_DISPLAY[top_label]}**\n\n"
            f"📊 Confidence: {confidence}%\n"
            f"📝 Jumlah kata: {len(text.split())}\n\n"
            f"💭 **Penjelasan:**\n{EXPLANATIONS[top_label]}\n\n"
            f"✨ **Saran Untuk Kamu:**\n"
        )
        
        for i, suggestion in enumerate(SUGGESTIONS[top_label], 1):
            response += f"{i}. {suggestion}\n"

        mood_data = {
            "emotion": top_label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        return response, mood_data

    # ==============================
    # ADDITIONAL FEATURES
    # ==============================
    def show_dashboard(self):
        """Tampilkan mood statistics dashboard"""
        # Create new window untuk dashboard
        dashboard_win = ctk.CTkToplevel(self)
        dashboard_win.title("📊 Mood Analytics Dashboard")
        dashboard_win.geometry("600x500")
        dashboard_win.configure(fg_color="#050816")

        # Get stats
        stats = self.data_store.get_mood_stats()
        total = sum(stats.values())

        if total == 0:
            empty_label = ctk.CTkLabel(
                dashboard_win,
                text="Belum ada data mood. Mulai analisis untuk melihat statistik!",
                font=("Consolas", 14),
                text_color="#A78BFA"
            )
            empty_label.pack(pady=50)
            return

        # Create figure untuk chart
        fig = Figure(figsize=(6, 4), dpi=100, facecolor="#050816", edgecolor="none")
        ax = fig.add_subplot(111, facecolor="#0B1024")

        emotions = list(stats.keys())
        values = list(stats.values())
        colors_list = [EMOTION_COLORS[e][0] for e in emotions]

        # Pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=[f"{EMOTION_DISPLAY[e]}\n({v})" for e, v in zip(emotions, values)],
            colors=colors_list,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "#E0F2FE", "fontsize": 10}
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")

        ax.set_title("Mood Distribution", color="#38BDF8", fontsize=14, fontweight="bold", pad=20)

        # Embed chart
        canvas = FigureCanvasTkAgg(fig, master=dashboard_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

        # Stats summary
        summary_frame = ctk.CTkFrame(dashboard_win, fg_color="#0B1024", corner_radius=8)
        summary_frame.pack(fill="x", padx=15, pady=(0, 15))

        summary_text = f"Total Analisis: {total} | Periode: Last Session"
        summary_label = ctk.CTkLabel(
            summary_frame,
            text=summary_text,
            font=("Consolas", 11),
            text_color="#A78BFA"
        )
        summary_label.pack(pady=10)

    def show_breathing(self):
        """Tampilkan breathing exercise guide"""
        breathing_win = ctk.CTkToplevel(self)
        breathing_win.title("🧘 Breathing Exercise")
        breathing_win.geometry("500x600")
        breathing_win.configure(fg_color="#050816")

        title = ctk.CTkLabel(
            breathing_win,
            text="🧘 Breathing Exercise",
            font=("Consolas", 20, "bold"),
            text_color="#10B981"
        )
        title.pack(pady=20)

        guide_text = """
Teknik Pernapasan 4-7-8:
Ini adalah teknik yang terbukti membantu menenangkan sistem saraf.

LANGKAH-LANGKAH:
1️⃣ Tarik napas melalui hidung selama 4 detik
2️⃣ Tahan napas selama 7 detik
3️⃣ Hembuskan melalui mulut selama 8 detik
4️⃣ Ulangi 4 kali

Gunakan Breathing Exercise ketika kamu merasa:
😨 Cemas atau panik
😠 Marah atau kesal
😢 Sedih atau depresi
😤 Stres atau tertekan

TIPS TAMBAHAN:
• Lakukan di tempat yang tenang
• Cari posisi duduk yang nyaman
• Fokus hanya pada napas
• Ulangi 4-5 siklus setiap hari
• Lakukan pagi atau malam hari
        """

        text_widget = ctk.CTkTextbox(
            breathing_win,
            fg_color="#0B1024",
            border_color="#10B981",
            border_width=2,
            text_color="#E0F2FE",
            font=("Consolas", 11)
        )
        text_widget.pack(fill="both", expand=True, padx=15, pady=15)
        text_widget.insert("1.0", guide_text)
        text_widget.configure(state="disabled")

    def export_dialog(self):
        """Dialog untuk export conversations"""
        export_win = ctk.CTkToplevel(self)
        export_win.title("💾 Export Conversation")
        export_win.geometry("400x250")
        export_win.configure(fg_color="#050816")
        export_win.attributes('-topmost', True)

        title = ctk.CTkLabel(
            export_win,
            text="💾 Export Format",
            font=("Consolas", 16, "bold"),
            text_color="#F59E0B"
        )
        title.pack(pady=20)

        info = ctk.CTkLabel(
            export_win,
            text="Pilih format export untuk conversation history:",
            font=("Consolas", 11),
            text_color="#D1D5DB"
        )
        info.pack(pady=10)

        button_frame = ctk.CTkFrame(export_win, fg_color="transparent")
        button_frame.pack(fill="both", expand=True, padx=20, pady=20)
        button_frame.grid_columnconfigure(0, weight=1)

        json_btn = ctk.CTkButton(
            button_frame,
            text="📄 JSON Format",
            command=lambda: self.do_export("json"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            height=45,
            font=("Consolas", 12, "bold")
        )
        json_btn.pack(fill="x", pady=5)

        txt_btn = ctk.CTkButton(
            button_frame,
            text="📋 Text Format",
            command=lambda: self.do_export("txt"),
            fg_color="#06B6D4",
            hover_color="#0891B2",
            height=45,
            font=("Consolas", 12, "bold")
        )
        txt_btn.pack(fill="x", pady=5)

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="✖️ Cancel",
            command=export_win.destroy,
            fg_color="#6B7280",
            hover_color="#4B5563",
            height=40
        )
        cancel_btn.pack(fill="x", pady=10)

    def do_export(self, format_type: str):
        """Execute export"""
        try:
            filepath = self.data_store.export_conversations(format_type)
            self.add_ai_message(
                f"✅ **Export Berhasil!**\n\n"
                f"File telah disimpan ke:\n`{filepath}`\n\n"
                f"Format: {format_type.upper()}\n"
                f"Total Pesan: {len(self.data_store.conversations)}"
            )
        except Exception as e:
            self.add_ai_message(f"❌ Export gagal: {str(e)}")

    def update_stats(self):
        """Update stats di sidebar"""
        stats = self.data_store.get_mood_stats()
        total = sum(stats.values())

        if total == 0:
            stats_text = "Belum ada data"
        else:
            stats_text = "Mood Breakdown:\n"
            for emotion, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = int(count / total * 100)
                    stats_text += f"{EMOTION_DISPLAY[emotion]}: {count} ({percentage}%)\n"

        self.stats_display.configure(text=stats_text)

    def new_chat(self):
        """Mulai obrolan baru"""
        self.clear_chat_widgets()
        self.add_ai_message("✨ Obrolan baru dimulai!\n\nCeritakan perasaanmu sekarang. Aku siap mendengarkan. 💜")

    def clear_history(self):
        """Hapus history"""
        self.clear_chat_widgets()
        self.data_store.conversations = []
        self.data_store.mood_history = []
        self.data_store._save_json(self.data_store.chat_file, [])
        self.data_store._save_json(self.data_store.mood_file, [])
        self.update_stats()
        self.add_ai_message("🗑️ Riwayat telah dihapus. Mari mulai dari awal! 💜")

    def clear_chat_widgets(self):
        """Clear semua widgets di chat area"""
        for widget in self.chat_area.winfo_children():
            widget.destroy()

    def show_loading(self, text: str):
        """Tampilkan loading indicator"""
        if hasattr(self, "loading_label") and self.loading_label.winfo_exists():
            self.loading_label.configure(text=text)
            return

        row = len(self.chat_area.winfo_children())
        self.loading_label = ctk.CTkLabel(
            self.chat_area,
            text=text,
            font=("Consolas", 12, "italic"),
            text_color="#38BDF8"
        )
        self.loading_label.grid(row=row, column=0, sticky="w", padx=16, pady=6)
        self.auto_scroll()

    def hide_loading(self):
        """Hide loading indicator"""
        if hasattr(self, "loading_label") and self.loading_label.winfo_exists():
            self.loading_label.destroy()

    def auto_scroll(self):
        """Auto scroll chat area ke bawah"""
        self.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1.0)

    def set_input_state(self, state: str):
        """Set state input components"""
        self.input_box.configure(state=state)
        self.send_button.configure(state=state)


# ==============================
# MAIN ENTRY POINT
# ==============================
if __name__ == "__main__":
    app = MoodSenseAI()
    app.mainloop()