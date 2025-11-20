# voice_age_checker_pretty.py
# 🎀 20代女性向けかわいいUI付き：あなたの声年齢診断（エンタメ用）

import tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st


# ========= カスタムCSS =========
CUTE_CSS = """
<style>
/* Google Fonts：かっちり × 柔らかの中間 */
@import url('https://fonts.googleapis.com/css2?family=Shippori+Antique+B1&family=Zen+Kaku+Gothic+New:wght@300;400;500&display=swap');

/* 背景（淡いミント） */
.stApp {
    background-color: #E6F6EE;
}

/* h1（タイトル）＝しっかり、上品、でも硬すぎない */
h1 {
    font-family: 'Shippori Antique B1', sans-serif !important;
    color: #2F6F63 !important;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: 0.02em;
}

/* h2（大見出し） */
h2 {
    font-family: 'Shippori Antique B1', sans-serif !important;
    color: #3D8578 !important;
    font-weight: 600;
    font-size: 1.7rem;
}

/* h3（小見出し） */
h3 {
    font-family: 'Shippori Antique B1', sans-serif !important;
    color: #4FA694 !important;
    font-weight: 500;
    font-size: 1.35rem;
}

/* 本文・説明・ラベル */
p, div, label, span {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    font-size: 1rem;
    color: #333;
}

/* ボタン */
button, .stButton>button {
    background-color: #76C7AF !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.3rem;
    font-family: 'Shippori Antique B1', sans-serif !important;
    font-weight: 600;
    font-size: 1rem;
    border: none;
}

/* ボタンHover */
.stButton>button:hover {
    background-color: #69B9A3 !important;
    transform: scale(1.03);
    transition: 0.15s ease-in-out;
}

/* 入力欄 */
input, select, textarea {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    border-radius: 8px !important;
}

/* ラジオ・セレクト */
.stRadio label, .stSelectbox label {
    font-family: 'Zen Kaku Gothic New', sans-serif !important;
    font-size: 1rem;
}
</style>
"""

# ========= 音声特徴量の抽出 =========
def extract_voice_features(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=None, mono=True)

    if np.max(np.abs(y)) < 1e-4:
        y = y + np.random.normal(0, 1e-4, size=len(y))

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_valid = f0[np.isfinite(f0)]
    if len(f0_valid) == 0:
        f0_mean, f0_std = 0.0, 0.0
    else:
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))

    return {
        "duration": float(duration),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "tempo": float(tempo),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
    }

# ========= 正規化 =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))

# ========= 声年齢推定 =========
def estimate_voice_age(features: dict, real_age: int, gender: str) -> dict:

    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_mean = features["f0_mean"]

    if gender == "女性":
        f0_young_min, f0_young_max = 200, 320
        f0_old_min, f0_old_max = 140, 220
    elif gender == "男性":
        f0_young_min, f0_young_max = 120, 220
        f0_old_min, f0_old_max = 80, 150
    else:
        f0_young_min, f0_young_max = 150, 260
        f0_old_min, f0_old_max = 100, 200

    pitch_youth = normalize(f0_mean, f0_old_min, f0_young_max)
    bright_youth = normalize(centroid_mean, 1000, 4000)
    tempo_youth = normalize(tempo, 60, 180)

    youth_score = (
        0.4*pitch_youth +
        0.35*bright_youth +
        0.25*tempo_youth
    )
    youth_score = float(np.clip(youth_score, 0, 1))

    MIN_AGE, MAX_AGE = 15, 80
    raw_voice_age = MIN_AGE + (1 - youth_score) * (MAX_AGE - MIN_AGE)

    voice_age = 0.6 * raw_voice_age + 0.4 * float(real_age)
    voice_age = float(np.clip(voice_age, MIN_AGE, MAX_AGE))

    diff = voice_age - real_age

    if diff <= -7:
        label = "実年齢よりかなり若く聞こえる声"
    elif diff <= -3:
        label = "実年齢より少し若く聞こえる声"
    elif diff < 3:
        label = "実年齢と近い“年相応”の声"
    elif diff < 7:
        label = "実年齢より少し落ち着いて聞こえる声"
    else:
        label = "実年齢よりかなり落ち着いて聞こえる声"

    return {
        "voice_age": voice_age,
        "real_age": float(real_age),
        "diff": diff,
        "label": label,
        "youth_score": youth_score,
        "pitch_youth": pitch_youth,
        "bright_youth": bright_youth,
        "tempo_youth": tempo_youth,
    }

# ========= レポート =========
def explain_result(result: dict, features: dict, gender: str) -> str:
    va = result["voice_age"]
    ra = result["real_age"]
    diff = result["diff"]
    label = result["label"]
    ys = result["youth_score"]

    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_mean = features["f0_mean"]

    lines = []
    lines.append("● 診断サマリー")
    lines.append(f"- 実年齢　　　　　　 : {ra:.0f} 歳")
    lines.append(f"- 推定 声年齢　　　　: {va:.0f} 歳")
    if diff >= 0:
        lines.append(f"- 差　　　　　　　　: +{diff:.1f} 歳（落ち着き寄り）")
    else:
        lines.append(f"- 差　　　　　　　　: {diff:.1f} 歳（若々しさ寄り）")
    lines.append(f"- 印象ラベル　　　　 : {label}")
    lines.append("")
    lines.append("● 若々しさスコア（0〜1）")
    lines.append(f"- 総合若々しさ : {ys:.2f}")
    lines.append("")

    return "\n".join(lines)

# ========= Streamlit UI =========
def main():

    # CSS記述のデザインの適用
    st.markdown(CUTE_CSS, unsafe_allow_html=True)


    st.set_page_config(page_title="あなたの声年齢診断", page_icon="🎂", layout="centered")
    st.markdown(CUTE_CSS, unsafe_allow_html=True)

    st.title("あなたの声年齢診断")

    st.markdown('<div class="pretty-card">', unsafe_allow_html=True)
    st.subheader("1. あなたの基本情報")

    col1, col2 = st.columns(2)
    with col1:
        real_age = st.number_input("実年齢", min_value=10, max_value=100, value=28)
    with col2:
        gender = st.selectbox("声の性別イメージ", ["女性", "男性", "その他・決めたくない"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pretty-card">', unsafe_allow_html=True)
    st.subheader("2. 声の録音")
    wav_audio_data = st.audio_input("🎤 録音してください")
    st.markdown("</div>", unsafe_allow_html=True)

    if wav_audio_data is None:
        st.info("音声を録音すると診断できます。")
        return

    # 一時ファイル
    audio_bytes = wav_audio_data.getbuffer()
    suffix = Path(wav_audio_data.name).suffix if wav_audio_data.name else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.audio(tmp_path)

    if st.button("声年齢を診断する"):
        with st.spinner("診断中…少しだけお待ちください"):
            features = extract_voice_features(tmp_path)
            result = estimate_voice_age(features, int(real_age), gender)
            report = explain_result(result, features, gender)

        st.markdown('<div class="pretty-card">', unsafe_allow_html=True)
        st.subheader("3. 診断結果")

        st.metric("推定 声年齢", f"{result['voice_age']:.0f} 歳")
        diff = result["diff"]
        if diff >= 0:
            st.metric("実年齢との差", f"+{diff:.1f} 歳")
        else:
            st.metric("実年齢との差", f"{diff:.1f} 歳")
        st.markdown(f"**印象：{result['label']}**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="pretty-card">', unsafe_allow_html=True)
        st.subheader("4. 詳細レポート")
        st.text(report)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
