# voice_age_checker.py
# 🎙 あなたの声年齢診断（エンタメ用）

import tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st

# ========= 音声特徴量の抽出 =========
def extract_voice_features(file_path: str) -> dict:
    """音声ファイルから簡易特徴量を抽出"""

    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 無音対策：完全無音だと一部特徴量が壊れるので微小ノイズ
    if np.max(np.abs(y)) < 1e-4:
        y = y + np.random.normal(0, 1e-4, size=len(y))

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]                 # 音量
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]  # ザラつき
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # 明るさ
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)    # テンポ

    # ピッチ（声の高さ）
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_valid = f0[np.isfinite(f0)]
    if len(f0_valid) == 0:
        f0_mean = 0.0
        f0_std = 0.0
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


# ========= スコアリング & 声年齢推定 =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def estimate_voice_age(features: dict, real_age: int, gender: str) -> dict:
    """
    声の特徴から「若々しさスコア」を作り、そこから声年齢を推定する（エンタメ用）

    基本の考え方：
      - 高めの声・明るい音色・少し速めのテンポ → 若く聞こえやすい
      - 低めの声・落ち着いた音色・ゆったりテンポ → 落ち着いて聞こえやすい
    """

    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_mean = features["f0_mean"]

    # 性別によって声の高さの想定レンジを変える
    if gender == "女性":
        f0_young_min, f0_young_max = 200, 320   # 若め
        f0_old_min, f0_old_max = 140, 220       # 落ち着きめ
    elif gender == "男性":
        f0_young_min, f0_young_max = 120, 220
        f0_old_min, f0_old_max = 80, 150
    else:  # その他・不明は中間くらい
        f0_young_min, f0_young_max = 150, 260
        f0_old_min, f0_old_max = 100, 200

    # 「若々しさ」っぽい指標を 0〜1 で作る
    pitch_youth = normalize(f0_mean, f0_old_min, f0_young_max)  # 高いほど若いと仮定
    bright_youth = normalize(centroid_mean, 1000, 4000)         # 明るさ
    tempo_youth = normalize(tempo, 60, 180)                     # 速めのテンポ

    # 重みづけして合成
    youth_score = (
        0.4 * pitch_youth +
        0.35 * bright_youth +
        0.25 * tempo_youth
    )
    youth_score = float(np.clip(youth_score, 0.0, 1.0))

    # 声年齢レンジ（だいたい15〜80歳くらいのイメージ）
    MIN_AGE, MAX_AGE = 15, 80

    # youth_score=1 → かなり若々しい → MIN_AGE 付近
    # youth_score=0 → 落ち着き強め        → MAX_AGE 付近
    raw_voice_age = MIN_AGE + (1.0 - youth_score) * (MAX_AGE - MIN_AGE)

    # 実年齢とも少しブレンドして、あまり極端になりすぎないようにする
    # （実年齢 40 の人が「声年齢 16歳」と出ないように、など）
    voice_age = 0.6 * raw_voice_age + 0.4 * float(real_age)

    # 常識的な範囲にクリップ
    voice_age = float(np.clip(voice_age, MIN_AGE, MAX_AGE))

    # 差分
    diff = voice_age - real_age

    # コメント用ラベル
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
    lines.append(f"- あなたの実年齢　　　: {ra:.0f} 歳")
    lines.append(f"- 推定される声年齢　　: {va:.0f} 歳")
    if diff >= 0:
        lines.append(f"- 差　　　　　　 　　: 実年齢より +{diff:.1f} 歳くらいに聞こえる傾向")
    else:
        lines.append(f"- 差　　　　　　 　　: 実年齢より {diff:.1f} 歳くらい若く聞こえる傾向")
    lines.append(f"- 声の印象ラベル　　　: {label}")
    lines.append("")

    lines.append("● 若々しさスコア（0〜1）")
    lines.append(f"- 総合若々しさ（youth_score） : {ys:.2f}")
    lines.append("  （高いほど“若く・軽やかに”聞こえやすい傾向を意味します）")
    lines.append("")

    lines.append("● 声のざっくり傾向")
    # 性別とピッチの簡易コメント
    if gender == "女性":
        if f0_mean > 260:
            lines.append("・声の高さはやや高めで、明るく元気な印象を与えやすいレンジです。")
        elif f0_mean < 180:
            lines.append("・声の高さはやや低めで、落ち着きや安心感を与えやすいレンジです。")
        else:
            lines.append("・声の高さは中くらいで、バランスの良い印象になりやすいレンジです。")
    elif gender == "男性":
        if f0_mean > 180:
            lines.append("・声の高さはやや高めで、軽やかさやフレンドリーさが出やすいレンジです。")
        elif f0_mean < 110:
            lines.append("・声の高さはやや低めで、落ち着きや重厚感が出やすいレンジです。")
        else:
            lines.append("・声の高さは中くらいで、柔らかさと落ち着きのバランスが良いレンジです。")
    else:
        lines.append("・声の高さは中庸〜やや個性寄りで、“あなたらしさ”が出やすいレンジです。")

    # 明るさ・テンポ
    if centroid_mean > 3000:
        lines.append("・声のトーン（明るさ）はやや高めで、元気・若々しさを感じやすい傾向があります。")
    elif centroid_mean < 1500:
        lines.append("・声のトーンは落ち着き寄りで、しっとり・大人っぽい印象になりやすいです。")
    else:
        lines.append("・声のトーンは中くらいで、明るすぎず落ち着きすぎず、状況に合わせて変えやすいタイプです。")

    if tempo > 130:
        lines.append("・話すテンポはやや速め寄りで、活発・テンション高めな印象を持たれやすいかもしれません。")
    elif tempo < 80:
        lines.append("・話すテンポはゆったりめで、落ち着いた・マイペースな印象を持たれやすいです。")
    else:
        lines.append("・話すテンポは標準的で、相手が聞き取りやすいスピードになりやすいです。")

    lines.append("")
    lines.append("※この診断は、声の特徴からざっくりと“若々しく聞こえるか / 落ち着いて聞こえるか”を推定するエンタメ用ツールです。")
    lines.append("※本当の年齢や見た目、性格とは必ずしも一致しませんので、“声の雰囲気を楽しむ”くらいの気持ちで使ってくださいね。")

    return "\n".join(lines)


# ========= Streamlit UI =========
def main():
    st.set_page_config(page_title="あなたの声年齢診断", page_icon="🎂", layout="centered")

    st.title("🎂 あなたの声年齢診断")
    st.caption("声の高さ・明るさ・テンポなどから、“声だけ聞いたときの印象年齢”をざっくり推定するエンタメ診断です。")

    st.markdown("---")
    st.subheader("1. あなたの情報を入力")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        real_age = st.number_input("あなたの実年齢（半角数字）", min_value=10, max_value=100, value=30, step=1)
    with col_info2:
        gender = st.selectbox("声の性別イメージ", ["女性", "男性", "その他・決めたくない"])

    st.markdown(
        """
        ※「声の性別イメージ」は、声の高さレンジの目安に使うだけなので、  
        実際の性自認や戸籍とは関係なく、**“この声はどっち寄りかな？”**くらいの感覚で選んでOKです。
        """
    )

    st.markdown("---")
    st.subheader("2. 声の録音")

    st.write("普段どおりの話し方で、30秒〜1分ほど話した音声がおすすめです。")

    wav_audio_data = st.audio_input(
        "🎤 録音を開始してください"
    )

    # 録音データまたはアップロードファイルが取得できていなければ終了
    if wav_audio_data is None:
        st.info("音声を録音すると診断が開始します。")
        return

    # 【✅ 処理ロジックの修正：UploadedFileオブジェクトとして処理】
    # st.audio_inputはUploadedFileオブジェクトを返す
    
    # 一時ファイルに保存
    tmp_path = None
    # 録音/アップロードされたデータは UploadedFile.getbuffer() でバイト列として取得
    audio_bytes = wav_audio_data.getbuffer() 
    
    # アップロード時のファイル名があればその拡張子を使う
    suffix = Path(wav_audio_data.name).suffix if wav_audio_data.name else ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # バイト列を一時ファイルに書き込む
        tmp.write(audio_bytes) 
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    # st.audio() には一時ファイルのパスを渡す
    st.audio(tmp_path) 

    
    if st.button("声年齢を診断する"):
        with st.spinner("声から“声年齢”を推定しています..."):
            try:
                features = extract_voice_features(tmp_path)
                result = estimate_voice_age(features, int(real_age), gender)
                report = explain_result(result, features, gender)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        st.markdown("---")
        st.subheader("3. 診断結果")

        st.metric("推定 声年齢", f"{result['voice_age']:.0f} 歳")
        diff = result["diff"]
        if diff >= 0:
            st.metric("実年齢との差", f"+{diff:.1f} 歳（落ち着き寄り）")
        else:
            st.metric("実年齢との差", f"{diff:.1f} 歳（若々しさ寄り）")

        st.markdown(f"**声の印象：{result['label']}**")

        st.markdown("---")
        st.subheader("4. 詳細レポート")
        st.text(report)

        st.markdown("---")
        st.caption(
            "※このツールは研究・医療・採用などの目的には使えません。"
            "あくまで“自分の声の雰囲気を知って楽しむための、お遊び診断”としてご利用ください。"
        )


if __name__ == "__main__":
    main()
