# voice_abnormal_checker.py
# 🎭 声でわかるあなたの変態度（アブノーマル度）
#
# ※ここでの「変態」は性的な意味ではなく、
#   「いい意味でのマニアックさ・こだわりの強さ・変人度」を
#   ゆるくいじるエンタメ診断です。

import tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st


# ========= 音声特徴量の抽出 =========
def extract_voice_features(file_path: str) -> dict:
    """音声ファイルから簡易特徴量を抽出"""

    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 無音対策
    if np.max(np.abs(y)) < 1e-4:
        y = y + np.random.normal(0, 1e-4, size=len(y))

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]                 # 音量
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]  # ザラつき・雑さ
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


# ========= スコアリング（声の性質） =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def score_voice_traits(features: dict) -> dict:
    """
    声の特徴からざっくり7つの指標を作る
    """
    rms_mean = features["rms_mean"]
    rms_std = features["rms_std"]
    zcr_mean = features["zcr_mean"]
    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_std = features["f0_std"]

    # 元気さ
    energy = 0.6 * normalize(rms_mean, 0.005, 0.05) + 0.4 * normalize(tempo, 60, 180)

    # 緊張感
    tension = (
        0.4 * normalize(zcr_mean, 0.01, 0.2)
        + 0.3 * normalize(rms_std, 0.0, 0.03)
        + 0.3 * normalize(centroid_mean, 1000, 4000)
    )

    # 安定感
    stability = 1.0 - 0.5 * normalize(f0_std, 0, 40) - 0.5 * normalize(rms_std, 0.0, 0.03)
    stability = float(np.clip(stability, 0.0, 1.0))

    # 明るさ
    brightness = normalize(centroid_mean, 1000, 4000)

    # ザラつき
    roughness = normalize(zcr_mean, 0.01, 0.2)

    # 抑揚
    pitch_var = normalize(f0_std, 0, 40)

    # テンポ
    tempo_index = normalize(tempo, 60, 180)

    return {
        "energy": energy,
        "tension": tension,
        "stability": stability,
        "brightness": brightness,
        "roughness": roughness,
        "pitch_var": pitch_var,
        "tempo_index": tempo_index,
    }


# ========= アブノーマル指数 & 変態タイプ判定 =========
def compute_abnormal_index(traits: dict) -> float:
    """
    “変態度（アブノーマル指数）”を 0〜100 のスコアで作る（完全エンタメ用）

    ざっくりと：
      - ザラつきがある（roughness高い）
      - 抑揚が激しい（pitch_var高い）
      - エネルギー高いのに安定感が低い
    などを「マニアックさ・クセ強さ」とみなす。
    """
    e = traits["energy"]
    s = traits["stability"]
    r = traits["roughness"]
    pv = traits["pitch_var"]

    # クセ強さ指標
    quirk = 0.4 * r + 0.4 * pv + 0.2 * (e * (1.0 - s))

    # 0〜1 → 0〜100
    score = float(np.clip(quirk, 0.0, 1.0) * 100.0)
    return score


def classify_type(score: float, traits: dict) -> dict:
    """
    アブノーマル指数 ＋ 声の性質から“変態タイプ”と“変態度の分野”を決める
    ※ここでの「変態」は「いい意味での変人・オタク・こだわり強め」のこと
    """
    e = traits["energy"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    t = traits["tension"]

    # ベースランク
    if score < 25:
        rank = "Lv.1：隠れ変態（ほぼノーマル）"
        rank_msg = (
            "普段はかなり“まとも側”に見られがちですが、"
            "ハマると一気に深堀りする“隠れマニア”タイプの気配があります。"
        )
    elif score < 50:
        rank = "Lv.2：ゆる変態（いい感じの変人）"
        rank_msg = (
            "日常会話の中に、ときどき“クセのあるこだわり”がにじむレベル。"
            "周りからは「ちょっと変わってて面白い人」と思われやすいゾーンです。"
        )
    elif score < 75:
        rank = "Lv.3：ガチ変態（マニアック勢）"
        rank_msg = (
            "何か一つでもハマる対象があると、相当ディープに掘ってしまう傾向。"
            "こだわりの強さは、立派な“変態（ほめ言葉）”クラスです。"
        )
    else:
        rank = "Lv.4：重症変態（こだわり職人）"
        rank_msg = (
            "もはや“その道の人”レベルのこだわりを発揮しがち。"
            "周囲からは「よくそこまでやるね…」と若干引かれつつも、"
            "同じ属性の人からはめちゃくちゃ愛されるタイプです。"
        )

    # タイプ分岐（音声特徴の組み合わせで決定）
    # 同時に「変態度の分野」も付与
    if pv > 0.7 and b > 0.5 and e > 0.6:
        type_name = "妄想クリエイター変態"
        type_desc = (
            "声の抑揚が大きく、明るさもあり、エネルギーも高め。"
            "頭の中でいろんなストーリーやシーンをこっそり妄想してそうな、"
            "“物語・世界観オタク” 系の変態気質です。\n"
            "アイデア出し・企画・創作活動との相性がかなり良いタイプ。"
        )
        domain_name = "創作・妄想系の変態度"
        domain_desc = (
            "小説・マンガ・ゲーム・ドラマ・設定資料集など、"
            "“世界観をつくる／読み解く系”のコンテンツで変態性が発揮されやすいタイプです。"
        )

    elif e > 0.7 and s > 0.6 and t < 0.5:
        type_name = "仕事ガチ勢・職人変態"
        type_desc = (
            "エネルギーと安定感の両方が高く、いい意味で“作業ガチ勢”の声です。"
            "一度スイッチが入ると、時間を忘れて没頭する職人タイプ。\n"
            "細部にこだわりすぎて気づいたら深夜、みたいな経験が多いかもしれません。"
        )
        domain_name = "仕事・スキル分野の変態度"
        domain_desc = (
            "仕事・資格勉強・クリエイティブ作業・スキル習得など、"
            "“成果物”が絡む領域で変態性が発揮されやすいタイプです。"
        )

    elif r > 0.7 and e > 0.5:
        type_name = "テンション暴走系変態"
        type_desc = (
            "声にザラつきや勢いがあり、ちょっと荒々しいテンションを感じます。"
            "盛り上がると一気に加速して周囲を巻き込むタイプの変人気質。\n"
            "ライブ・イベント・飲み会など、“現場ノリ”と相性がいいタイプです。"
        )
        domain_name = "遊び・イベント分野の変態度"
        domain_desc = (
            "飲み会・ライブ・フェス・イベント・旅行など、"
            "“その場のノリや体験”で変態性が全開になりやすいタイプです。"
        )

    elif e < 0.4 and s > 0.6 and pv < 0.5:
        type_name = "観察者タイプの静かなる変態"
        type_desc = (
            "声は落ち着いていて安定しており、抑揚も控えめ。"
            "表に出るより、じっと人や状況を観察している“裏方マニア”系です。\n"
            "人間観察・考察・分析系コンテンツと相性の良い変態タイプ。"
        )
        domain_name = "観察・分析分野の変態度"
        domain_desc = (
            "人間観察・SNSの動向・データ・歴史・作品解説など、"
            "“じっくり見て考える系”の領域で変態性がにじみ出るタイプです。"
        )

    elif b < 0.4 and s < 0.5:
        type_name = "沼に沈みがちなディープ変態"
        type_desc = (
            "声のトーンはやや暗めで、安定感も少し揺れ気味。"
            "一度ハマると、その世界からなかなか戻ってこない“沼体質”の可能性が。\n"
            "推し活・ゲーム・趣味など、特定ジャンルへの没入度はかなり高そうです。"
        )
        domain_name = "沼・収集・推し活分野の変態度"
        domain_desc = (
            "推し活・コレクション・ゲーム・特定ジャンルの追いかけなど、"
            "“一点集中で深く潜る系”の領域で変態性が最大化しやすいタイプです。"
        )

    else:
        type_name = "バランス型・日常に溶け込む変態"
        type_desc = (
            "声の特徴に大きな極端さはなく、日常生活では“普通の人”として溶け込むタイプ。"
            "ただし、一部のテーマだけ異常に詳しい“隠しオタク”要素を内包している可能性が高いです。"
        )
        domain_name = "日常生活全般にじわっと滲む変態度"
        domain_desc = (
            "仕事・趣味・人間関係・日常会話など、特定のジャンルに限らず、"
            "ふとした瞬間のこだわりや豆知識として変態性がにじみ出るタイプです。"
        )

    return {
        "rank": rank,
        "rank_msg": rank_msg,
        "type_name": type_name,
        "type_desc": type_desc,
        "domain_name": domain_name,
        "domain_desc": domain_desc,
    }


def explain_traits(traits: dict) -> str:
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    lines = []
    lines.append("● 声のざっくり性質（0〜1）")
    lines.append(f"- 元気さ（energy）      : {e:.2f}")
    lines.append(f"- 緊張感（tension）     : {t:.2f}")
    lines.append(f"- 安定感（stability）   : {s:.2f}")
    lines.append(f"- 明るさ（brightness）  : {b:.2f}")
    lines.append(f"- ザラつき（roughness） : {r:.2f}")
    lines.append(f"- 抑揚（pitch_var）      : {pv:.2f}")
    lines.append(f"- テンポ（tempo_index） : {ti:.2f}")
    lines.append("")

    lines.append("● 変態度に関わっていそうなポイント")
    if r > 0.6:
        lines.append("・声に少し“荒さ・生っぽさ”があり、勢いやクセの強さがにじみ出ています。")
    if pv > 0.6:
        lines.append("・抑揚が大きく、感情の振れ幅やリアクションが豊かに出やすいタイプです。")
    if e > 0.7:
        lines.append("・エネルギー高めで、“テンション上がると止まらない”モードに入りやすそうです。")
    if s < 0.4:
        lines.append("・安定感はやや低めで、心の中の揺らぎやこだわりが声に反映されやすいかもしれません。")

    if len(lines) == 2:
        lines.append("・大きな偏りはなく、“バランス型の変人候補”といった感じです。")

    lines.append("")
    lines.append("※この結果は、声の特徴から“それっぽく遊ぶための冗談診断”です。本気にしすぎず、ネタとして楽しんでください。")

    return "\n".join(lines)


# ========= Streamlit UI =========
def main():
    st.set_page_config(page_title="声でわかるあなたの変態度", page_icon="🦄", layout="centered")

    st.title("🦄 声でわかるあなたの変態度（アブノーマル度）")
    st.caption("※ここでの“変態”は、いい意味でのマニアックさ・こだわりオタク度をいじるエンタメ診断です。")

    st.markdown("---")
    st.subheader("1. 声を録音")

    st.write(
        "普段どおりの話し方で、20〜30秒くらいしゃべった音声をアップロードしてください。\n"
        "最近ハマっているものの話や、好きなことについて話してもらうと、結果がそれっぽくなります。"
    )

    wav_audio = st.audio_input("🎤 録音ボタンを押して話してください")

    if wav_audio is None:
        st.info("録音が完了すると診断できます。")
        return

    # 一時ファイルに保存
    audio_bytes = wav_audio.getbuffer()
    suffix = Path(wav_audio.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(audio_path)

    if st.button("変態度を診断する"):
        with st.spinner("あなたの“いい意味での変態度”を計測中..."):
            try:
                features = extract_voice_features(audio_path)
                traits = score_voice_traits(features)
                abnormal_score = compute_abnormal_index(traits)
                type_info = classify_type(abnormal_score, traits)
                trait_text = explain_traits(traits)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        st.markdown("---")
        st.subheader("2. 診断結果")

        st.metric("アブノーマル指数（変態度）", f"{abnormal_score:.1f} / 100")
        st.markdown(f"**ランク：{type_info['rank']}**")
        st.write(type_info["rank_msg"])

        st.markdown("---")
        st.subheader("3. あなたの変態タイプ")

        st.markdown(f"### 🧬 タイプ：{type_info['type_name']}")
        st.write(type_info["type_desc"])

        st.markdown("---")
        st.subheader("4. 変態度の分野")

        st.markdown(f"**分野：{type_info['domain_name']}**")
        st.write(type_info["domain_desc"])

        st.markdown("---")
        st.subheader("5. 声の特徴 詳細")
        st.text(trait_text)

        st.markdown("---")
        st.caption(
            "※この診断はジョーク用です。実際の性格・人格・嗜好を科学的に保証するものではありません。\n"
            "　“自分の変人っぽいところを笑いながら眺める”くらいの感覚で楽しんでください。"
        )


if __name__ == "__main__":
    main()
