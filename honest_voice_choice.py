# honest_voice_choice.py
# 🎙 声でわかるあなたの本音（エンタメ用）
#
# 使い方：
# 1. 「今回のテーマ」（例：今日のランチ）を入力
# 2. 選択肢A/B（例：ハンバーグ / パスタ）を入力
# 3. それぞれについて
#    「私が食べたいのはハンバーグです」
#    「私が食べたいのはパスタです」
#    などと喋った音声を録音し、アップロード
# 4. 声の“緊張・安定・エネルギー”から、どちらが本音ぽいかを
#    ざっくり診断します（※あくまでジョーク・自己観察用）

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
    """音声ファイルから簡易特徴量を抽出"""

    # sr=None で元レートのまま読み込み
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 完全無音だと固まることがあるので、微小ノイズを足す
    if np.max(np.abs(y)) < 1e-4:
        y = y + np.random.normal(0, 1e-4, size=len(y))

    # 基本特徴量
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]               # 音量
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]  # ザラザラ感・ノイズ感
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # 明るさ
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # ざっくりテンポ

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


# ========= スコアリング（緊張・安定・エネルギー・“本音度”） =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def score_from_features(features: dict) -> dict:
    """
    特徴量から 0〜1 のスコアを作成（ざっくりのニュアンス）

    energy   : 声の元気さ・外向き度
    tension  : 緊張・ぎこちなさ
    stability: 安定感・落ち着き
    honesty  : “本音っぽさ”指標（※完全にエンタメ用）
    """

    rms_mean = features["rms_mean"]
    rms_std = features["rms_std"]
    zcr_mean = features["zcr_mean"]
    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_std = features["f0_std"]

    # 元気さ
    energy = 0.6 * normalize(rms_mean, 0.005, 0.05) + 0.4 * normalize(tempo, 60, 180)

    # 緊張感：ノイズ感 + 音量の揺れ + （少し明るさ）
    tension = (
        0.4 * normalize(zcr_mean, 0.01, 0.2)
        + 0.3 * normalize(rms_std, 0.0, 0.03)
        + 0.3 * normalize(centroid_mean, 1000, 4000)
    )

    # 安定感：ピッチのブレ & 音量のブレが小さいほど高い
    stability = 1.0 - 0.5 * normalize(f0_std, 0, 40) - 0.5 * normalize(rms_std, 0.0, 0.03)
    stability = float(np.clip(stability, 0.0, 1.0))

    # “心地よさ”：中くらいのエネルギーが一番ラク、という仮定
    comfort = 1.0 - abs(energy - 0.5) * 2.0
    comfort = float(np.clip(comfort, 0.0, 1.0))

    # “本音度” 仮の定義：
    #  - 安定感が高い
    #  - 緊張感が低い
    #  - 自分にとってラクなエネルギー感（無理してテンション上げすぎ・落としすぎではない）
    honesty = 0.5 * stability + 0.3 * (1 - tension) + 0.2 * comfort
    honesty = float(np.clip(honesty, 0.0, 1.0))

    return {
        "energy": energy,
        "tension": tension,
        "stability": stability,
        "honesty": honesty,
    }


# ========= 解説メッセージ生成 =========
def explain_single(label: str, scores: dict) -> str:
    """
    各選択肢ごとの解説文（A案 / B案 それぞれ用）
    """
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]
    h = scores["honesty"]

    lines = []

    lines.append(f"【{label} と言ったときの声の傾向】")
    lines.append(f"- 本音っぽさ（honesty）  : {h:.2f}")
    lines.append(f"- 元気さ（energy）        : {e:.2f}")
    lines.append(f"- 緊張感（tension）       : {t:.2f}")
    lines.append(f"- 安定感（stability）     : {s:.2f}")
    lines.append("")

    # honesty のざっくりコメント
    if h > 0.7:
        lines.append("▶ 声の揺れが少なく、テンションも無理している感じがあまりないので、かなり“本音寄り”に聞こえます。")
    elif h > 0.5:
        lines.append("▶ わりと自然なトーンですが、少しだけ緊張や迷いも混じっていそうな雰囲気です。")
    else:
        lines.append("▶ 声の安定感やラクさがやや低めで、“ちょっと無理して言っている”ようにも聞こえるゾーンです。")

    # tension / stability による補足
    if t > 0.7:
        lines.append("・やや声に力が入りやすく、早口・強めのトーンになりやすい可能性があります。")
    elif t < 0.3:
        lines.append("・緊張感は少なめで、割とリラックスした状態で話せていそうです。")

    if s > 0.7:
        lines.append("・声の揺れが少なく、聞き手に“落ち着き”や“自信”を感じさせやすい状態です。")
    elif s < 0.3:
        lines.append("・声の揺れやブレがやや大きめで、心の中の揺らぎも少し反映されているかもしれません。")

    return "\n".join(lines)


def compare_honesty(label_a: str, scores_a: dict, label_b: str, scores_b: dict) -> str:
    """
    2つの選択肢の“本音度”を比較してコメント生成
    """
    ha = scores_a["honesty"]
    hb = scores_b["honesty"]

    if abs(ha - hb) < 0.07:
        # ほぼ同じくらい
        msg = (
            f"【結果】\n"
            f"・{label_a} を選んだときの本音度   : {ha:.2f}\n"
            f"・{label_b} を選んだときの本音度   : {hb:.2f}\n\n"
            "差はかなり僅差です。どちらも“そこそこ本音寄り”か、"
            "もしくは“どっちにしようか本気で揺れている”状態かもしれません。\n"
            "自分でもう一度心の中でつぶやいてみて、しっくりくる方を選んでみてください。"
        )
    elif ha > hb:
        msg = (
            f"【結果】\n"
            f"・{label_a} を選んだときの本音度   : {ha:.2f}\n"
            f"・{label_b} を選んだときの本音度   : {hb:.2f}\n\n"
            f"声の状態だけを見ると、**「{label_a}」と言ったときの方が本音に近い** 可能性が高そうです。\n"
            "よりラクで自然なトーンが出ている方を“本音寄り”と仮定した結果になっています。"
        )
    else:
        msg = (
            f"【結果】\n"
            f"・{label_a} を選んだときの本音度   : {ha:.2f}\n"
            f"・{label_b} を選んだときの本音度   : {hb:.2f}\n\n"
            f"声の状態だけを見ると、**「{label_b}」と言ったときの方が本音に近い** 可能性が高そうです。\n"
            "よりラクで自然なトーンが出ている方を“本音寄り”と仮定した結果になっています。"
        )

    msg += (
        "\n\n※この診断は、声の緊張や安定感から“なんとなくの本音度”を遊び感覚で推定しているだけで、"
        "実際の嘘・本音を科学的に判定するものではありません。"
    )

    return msg


# ========= Streamlit UI =========
def main():


    # CSS記述のデザインの適用
    st.markdown(CUTE_CSS, unsafe_allow_html=True)

    st.set_page_config(page_title="声でわかるあなたの本音", page_icon="🫧", layout="centered")

    st.title("🫧 声でわかるあなたの本音")

    # ======== 説明文（テーマ入力の代わりに追加） ========
    st.markdown(
        """
### 📘 このツールについて

このアプリでは **2つの声を録音してもらうだけで、  
あなたの“本音に近い方”をざっくり判定**します。

---

### 🔊 使い方の例

たとえば、晩ご飯を **ハンバーグ** にするか **パスタ** にするか迷っている場合は…

- **Aの音声**：「私が本当に食べたいのは **ハンバーグ** です」  
- **Bの音声**：「私が本当に食べたいのは **パスタ** です」

この2つを録音し、アップロードしてください。

もちろん、  
**晩ご飯・今日の予定・恋愛・趣味・買い物・転職など、食べ物以外のテーマでもOK！**

※あくまでジョーク・自己観察用の診断です。本当の嘘発見や心理分析ではありません。

---

        """
    )

    st.markdown("---")
    st.subheader("1. 音声をアップロード")

    # ===== 音声アップロード =====
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"### A の音声")
        st.caption("例：「私が本当に選びたいのはハンバーグです」")
        wav_audio_a = st.audio_input("🎤 A の音声を録音してください")

    with col_b:
        st.markdown(f"### B の音声")
        st.caption("例：「私が本当に選びたいのはパスタです」")
        wav_audio_b = st.audio_input("🎤 B の音声を録音してください")

    
    # ===== 一時保存 =====
    if wav_audio_a is None or wav_audio_b is None:
        st.info("AとBの音声を両方録音すると診断できます。")
        return

    # Aの音声の一時保存
    audio_bytes_a = wav_audio_a.getbuffer()
    suffix_a = Path(wav_audio_a.name).suffix if wav_audio_a.name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a) as tmp_a:
        tmp_a.write(audio_bytes_a)
        path_a = tmp_a.name

    # Bの音声の一時保存
    audio_bytes_b = wav_audio_b.getbuffer()
    suffix_b = Path(wav_audio_b.name).suffix if wav_audio_b.name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b) as tmp_b:
        tmp_b.write(audio_bytes_b)
        path_b = tmp_b.name

    st.markdown("---")
    st.subheader("2. 音声のプレビュー")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("▶ A の音声")
        st.audio(path_a)
    with col_p2:
        st.markdown("▶ B の音声")
        st.audio(path_b)

    # ===== 診断ボタン =====
    if st.button("本音度を診断する"):
        with st.spinner("声から“本音度”をざっくり計算しています..."):
            try:
                feat_a = extract_voice_features(path_a)
                feat_b = extract_voice_features(path_b)

                scores_a = score_from_features(feat_a)
                scores_b = score_from_features(feat_b)

                # ラベルは「A」「B」に固定
                result_text = compare_honesty("A", scores_a, "B", scores_b)
                explain_a = explain_single("A", scores_a)
                explain_b = explain_single("B", scores_b)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        st.markdown("---")
        st.subheader("3. 診断結果")

        st.text(result_text)

        st.markdown("---")
        st.subheader("4. 詳細レポート")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### A の声レポート")
            st.text(explain_a)
        with col_r2:
            st.markdown("#### B の声レポート")
            st.text(explain_b)

        st.markdown("---")
        st.caption(
            "※このツールは、声の“緊張・安定・ラクさ”などから本音っぽさを遊び感覚で推定するエンタメ診断です。\n"
            "大切な判断は、ご自身の気持ちと状況を踏まえて決めてくださいね。"
        )
    else:
        st.info("両方の音声ファイルをアップロードすると診断ボタンが表示されます。")


if __name__ == "__main__":
    main()
