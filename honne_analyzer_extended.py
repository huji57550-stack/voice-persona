"""
声＋音声から『本音っぽい食べ物／行きたい場所／したいこと』を推定する
エンタメ用ツール（Streamlit UI 内蔵版・Gemini + ローカル音声認識版）

★ このファイル単独で完結
★ 「streamlit run src/honne_analyzer_extended/honne_analyzer_extended.py」で起動できます

構成：
- アップロード音声を SpeechRecognition + Google Web Speech API で文字起こし（ローカルライブラリ）
- librosa で音声特徴量を抽出
- 推定ムード・ドメイン・本音候補をまとめて Gemini (gemini-pro) で
  「本音っぽい文章」を生成
"""

import os
import re
import random
import io
import tempfile

import librosa
import numpy as np
from dataclasses import dataclass, asdict

import speech_recognition as sr
from pydub import AudioSegment

# ====== Gemini（Google Generative AI）設定 ======
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY 環境変数が設定されていません。\n"
        "PowerShell などで  $env:GOOGLE_API_KEY = \"AIza...\"  を実行してください。"
    )

genai.configure(api_key=GOOGLE_API_KEY)

# 音声を直接渡さず、テキスト生成専用の gemini-pro を使う
GEMINI_MODEL = genai.GenerativeModel("gemini-pro")


# ======================================
# 結果データ
# ======================================

@dataclass
class HonneResult:
    domain: str          # "food" / "place" / "action"
    detected_term: str   # テキストから見つけたワード
    honne_term: str      # 推定本音キーワード（食べ物/場所/行動）
    mood_label: str      # 日本語ムードラベル
    message: str         # Gemini が生成した本音文章
    transcript: str      # 文字起こし結果


# --------------------------------------
# 候補データ
# --------------------------------------

FOODS = {
    "comfort": ["オムライス", "カレーライス", "親子丼", "ラーメン", "牛丼"],
    "junk": ["ハンバーガー", "唐揚げ定食", "ピザ", "ポテトフライ", "からあげクン"],
    "healthy": ["サラダボウル", "グリルチキン", "雑穀ごはん定食", "湯豆腐", "具だくさん味噌汁"],
    "sweet": ["プリン", "シュークリーム", "チーズケーキ", "パンケーキ", "パフェ"],
    "homey": ["肉じゃが", "しょうが焼き", "焼き魚定食", "煮物", "味噌汁とごはん"],
}

PLACES = {
    "sea": ["静かな海辺", "南の島のビーチ", "夕暮れの海岸"],
    "mountain": ["森の中の山小屋", "高原のキャンプ場", "星が見える山頂"],
    "onsen": ["山奥の温泉宿", "貸切露天風呂の旅館", "ひなびた温泉街"],
    "home": ["自宅のソファ", "自分のベッド", "こたつの中"],
    "city": ["お気に入りのカフェ", "映画館", "繁華街"],
}

ACTIONS = {
    "rest": ["ひたすら寝る", "ごろごろする", "布団でだらける"],
    "play": ["友だちと遊ぶ", "飲みに行く", "カラオケで歌う"],
    "nothing": ["予定ゼロで過ごす", "誰とも会わない", "完全オフにする"],
    "growth": ["勉強する", "読みかけの本を読む", "目標整理をする"],
    "care": ["マッサージに行く", "ゆっくり風呂に入る", "散歩して頭を整理する"],
}

FOOD_WORDS = ["ハンバーグ", "オムライス", "ラーメン", "そば", "うどん", "ピザ", "サラダ", "唐揚げ"]
PLACE_WORDS = ["海", "山", "温泉", "家", "カフェ", "映画館"]
ACTION_WORDS = ["寝たい", "遊びたい", "休みたい", "何もしたくない", "頑張りたい"]


# ======================================
# ローカル音声認識（Google Web Speech API 経由）
# ======================================

def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """
    SpeechRecognition + Google Web Speech API を使って日本語文字起こし。
    ※ Google側の無料エンドポイントを使うので、GOOGLE_API_KEY は不要。
    """
    # 1) アップロードされたバイト列を一時ファイルに変換（wavに統一）
    #    mp3/m4a の場合は pydub + ffmpeg が必要
    try:
        # mime_type からフォーマット推定（例: "audio/mpeg" -> "mp3"）
        fmt = mime_type.split("/")[-1]
        if fmt == "mpeg":
            fmt = "mp3"

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            audio_segment.export(wav_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"音声ファイルの変換に失敗しました（ffmpeg が必要な場合があります）：{e}")

    # 2) SpeechRecognition で wav を読み込んで Google に送る
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)

    try:
        text = r.recognize_google(audio_data, language="ja-JP")
        return text.strip()
    except sr.UnknownValueError:
        raise RuntimeError("音声がうまく聞き取れませんでした。もう少しはっきり話してみてください。")
    except sr.RequestError as e:
        raise RuntimeError(f"音声認識サービスへのアクセスに失敗しました: {e}")


# ======================================
# 音声解析まわり（librosa）
# ======================================

def load_audio_bytes(b: bytes, sr=22050):
    """アップロードされた音声のバイト列を読み込む"""
    y, sr = librosa.load(io.BytesIO(b), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y, sr


def extract_features(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    loudness = float(np.mean(rms))

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    voiced = f0[voiced_flag]
    pitch = float(np.nanmean(voiced)) if voiced.size > 0 else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    noisiness = float(np.mean(zcr))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    return {"loudness": loudness, "pitch": pitch, "noisiness": noisiness, "tempo": tempo}


def infer_mood(f):
    l, t, n = f["loudness"], f["tempo"], f["noisiness"]

    if l > 0.08 and t > 120:
        return "GENKI"
    if l < 0.03 and t < 80:
        return "TSUKARE"
    if n > 0.12 and t > 100:
        return "STRESS"
    return "CALM"


# ======================================
# ドメイン判定
# ======================================

def detect_domain(text):
    if "食べ" in text:
        return "food"
    if "行き" in text:
        return "place"
    if any(k in text for k in ["したい", "したく", "休みたい", "遊びたい", "寝たい"]):
        return "action"
    # デフォルトは food にしておく
    return "food"


def detect_word(text, words):
    for w in words:
        if w in text:
            return w
    return None


# ======================================
# 本音候補（キーワード）生成（ルールベース）
# ======================================

def choose_food(word, mood):
    if mood == "GENKI":
        c = FOODS["junk"] + FOODS["comfort"]
    elif mood == "TSUKARE":
        c = FOODS["comfort"] + FOODS["sweet"]
    elif mood == "STRESS":
        c = FOODS["junk"] + FOODS["sweet"]
    else:
        c = FOODS["healthy"] + FOODS["homey"]

    if word:
        c = [x for x in c if word not in x] or c

    return random.choice(c)


def choose_place(word, mood):
    if mood in ("TSUKARE", "STRESS"):
        c = PLACES["onsen"] + PLACES["home"]
    elif mood == "GENKI":
        c = PLACES["sea"] + PLACES["city"]
    else:
        c = PLACES["mountain"] + PLACES["home"]

    if word:
        c = [x for x in c if word not in x] or c

    return random.choice(c)


def choose_action(word, mood):
    if mood == "TSUKARE":
        c = ACTIONS["rest"] + ACTIONS["nothing"]
    elif mood == "STRESS":
        c = ACTIONS["care"] + ACTIONS["rest"]
    elif mood == "GENKI":
        c = ACTIONS["play"] + ACTIONS["growth"]
    else:
        c = ACTIONS["growth"] + ACTIONS["care"]

    return random.choice(c)


# ======================================
# Gemini で「本音文章」を生成
# ======================================

def generate_honne_message(
    domain: str,
    transcript: str,
    honne_term: str,
    mood_jp: str,
    features: dict,
) -> str:
    """
    文字起こしテキスト＋ムード＋音声特徴＋本音キーワードを渡して、
    Gemini (gemini-pro) に「本音っぽい文章」を作ってもらう。
    """
    feature_str = (
        f"平均音量(loudness)={features['loudness']:.3f}, "
        f"テンポ(tempo)={features['tempo']:.1f}, "
        f"ノイズ指標(noisiness)={features['noisiness']:.3f}"
    )

    domain_jp = {
        "food": "食べ物",
        "place": "行きたい場所",
        "action": "したいこと",
    }.get(domain, "その他")

    prompt = f"""
あなたは、声の雰囲気から“本音”をそれっぽく言語化する、
少しおちゃめな日本語カウンセラーです。
ただし、あくまでエンタメ用途で、ユーザーを傷つけない・不安にさせない表現にしてください。

トーンの条件：
- 上から目線や説教口調は避ける
- フランクで、友だちに話しかけるような文体で
- 文章は 1〜3 文程度
- 最初の1文で本音をズバッと、後ろの文で少し優しくフォロー
- 出力は日本語のみ

[入力情報]
- ドメイン: {domain_jp}
- 推定ムード: {mood_jp}
- 音声特徴: {feature_str}
- 文字起こしテキスト: 「{transcript}」
- 本音候補キーワード: 「{honne_term}」

[やってほしいこと]
上の情報から、
「ユーザーが口では言ってないけど、心のどこかで思っていそうな本音」を
日本語で 1〜3 文の短いテキストとして出力してください。

・{domain_jp}に関する本音として、必ず「{honne_term}」に触れてください。
・ムード「{mood_jp}」を少し匂わせる表現にしてください。
・診断結果の宣言や注意書き（例：これはフィクションです 等）は書かないでください。
・絵文字は使っても使わなくてもOKですが、多用しすぎないでください。
"""

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        text = response.text or ""
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini での本音生成に失敗しました: {e}") from e


# ======================================
# メイン処理
# ======================================

def analyze_honne(audio_bytes: bytes, mime_type: str) -> HonneResult:
    # 1) 文字起こし（ローカル＋Google Web Speech）
    transcript = transcribe_audio(audio_bytes, mime_type=mime_type)

    # 2) 音声特徴抽出
    y, sr = load_audio_bytes(audio_bytes)
    f = extract_features(y, sr)
    mood = infer_mood(f)

    mood_jp = {
        "GENKI": "テンション高めモード",
        "TSUKARE": "おつかれモード",
        "STRESS": "ストレス気味モード",
        "CALM": "落ち着きモード",
    }[mood]

    # 3) ドメイン判定
    domain = detect_domain(transcript)

    # 4) 本音候補キーワード（ルールベース）選択
    if domain == "food":
        word = detect_word(transcript, FOOD_WORDS)
        honne = choose_food(word, mood)

    elif domain == "place":
        word = detect_word(transcript, PLACE_WORDS)
        honne = choose_place(word, mood)

    else:
        word = detect_word(transcript, ACTION_WORDS)
        honne = choose_action(word, mood)

    # 5) Gemini で本音文章生成
    message = generate_honne_message(domain, transcript, honne, mood_jp, f)

    return HonneResult(
        domain=domain,
        detected_term=word or "（特定できず）",
        honne_term=honne,
        mood_label=mood_jp,
        message=message,
        transcript=transcript,
    )


# ======================================
# ★ Streamlit UI（このファイルだけで動く）
# ======================================

def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="声から本音分析（ネタ・Gemini版）", page_icon="🎙")

    st.title("🎙 声から本音分析ラボ（Gemini版・ネタ）")
    st.caption("声の雰囲気＋セリフから“それっぽい本音”を Gemini でつくるエンタメツールです。")

    st.markdown("### 1. 音声ファイルをアップロード")
    audio = st.file_uploader("wav / mp3 / m4a に対応（ブラウザ依存）", type=["wav", "mp3", "m4a"])

    st.markdown("### 2. 本音分析を実行")
    if st.button("音声から本音を分析する"):
        st.write("✅ ボタンが押されました")

        if audio is None:
            st.error("音声ファイルを先にアップしてください")
            return

        audio_bytes = audio.read()
        mime_type = audio.type or "audio/wav"
        st.write(f"✅ 音声バイト長: {len(audio_bytes)} / MIME: {mime_type}")

        try:
            with st.spinner("文字起こし＆音声分析＆本音生成 中…"):
                result = analyze_honne(audio_bytes, mime_type=mime_type)
        except Exception as e:
            st.error("エラーが発生しました。詳細は下をご確認ください。")
            st.exception(e)
            return

        st.markdown("## 🔊 文字起こし結果")
        st.write(result.transcript)

        st.markdown("## 🔍 本音の推定結果")
        st.markdown(f"### **{result.message.replace(chr(10), '  \n')}**")

        st.markdown("### 内部サマリー")
        st.json(asdict(result))

        st.info("※完全にエンタメ用フィクションです。診断ではありません。気軽に楽しんでください。")


# ======================================
# Streamlit 実行
# ======================================

if __name__ == "__main__":
    run_streamlit()
