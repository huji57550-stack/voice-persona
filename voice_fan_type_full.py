# voice_fan_type_full.py
# 🎧 あなたの声がささる人

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
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

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


# ========= スコアリング =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def score_voice_traits(features: dict) -> dict:
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


# ========= 「あなたの声が好きな人」タイプ判定（＋体格＋恋愛シチュ） =========
def classify_fan_types(traits: dict, gender: str) -> dict:
    """
    性別に応じて“あなたの声が刺さりやすい相手像”を最適化して返す関数
    """

    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    # 男女別ターゲットの呼び方
    if gender == "男性":
        target = "女性"
    else:
        target = "男性"

    # ===== メインタイプ =====
    if e > 0.7 and b > 0.6 and pv > 0.6:
        main_title = f"テンションを上げたい“陽キャ寄り”の{target}"
        if gender == "男性":
            main_desc = (
                "あなたの明るく抑揚のある声は、テンションを上げたい女性に強く響きます。\n"
                "・ノリのいい会話が好きな女性\n"
                "・明るい雰囲気の男性に惹かれる女性\n"
                "・落ち込んだときに元気をくれる存在を求めている女性\n"
                "そんな“陽キャ気質・元気をもらいたい女性”があなたの声にハマりやすい傾向があります。"
            )
        else:
            main_desc = (
                "あなたの明るく抑揚のある声は、前向きな気分になりたい男性に刺さりやすいです。\n"
                "・楽しい雰囲気の女性が好きな男性\n"
                "・リアクションを大事にする男性\n"
                "・テンションを上げたいときにあなたの声を好む男性\n"
                "そんな“陽キャ寄りな男性”に好まれやすいタイプです。"
            )

    elif e < 0.4 and s > 0.6 and b < 0.5:
        main_title = f"せかされるのが苦手な“おだやか系”の{target}"
        if gender == "男性":
            main_desc = (
                "あなたの落ち着いた安定した声は、穏やかさを求める女性に深い安心感を与えます。\n"
                "・マイペースで過ごしたい女性\n"
                "・穏やかな男性の声に癒される女性\n"
                "・せかされるのが苦手な女性\n"
                "そんな“ゆったりペースの女性”に刺さりやすい声です。"
            )
        else:
            main_desc = (
                "あなたの落ち着いた声は、プレッシャーや刺激を避けたい男性に心地よく響きます。\n"
                "・静かな環境を好む男性\n"
                "・安心感のある女性に惹かれる男性\n"
                "・優しく話してくれる相手を好む男性\n"
                "そんな“おだやかさ重視の男性”があなたの声を好みやすい傾向があります。"
            )

    elif s > 0.7 and t < 0.5:
        main_title = f"安心感を求める“ちょっと不安気味な{target}”"
        if gender == "男性":
            main_desc = (
                "あなたの安定した声は、不安を感じやすい女性にとって“癒し枠”になりやすいです。\n"
                "・緊張しやすい女性\n"
                "・優しく聞いてくれる男性に惹かれる女性\n"
                "・落ち着いたトーンに安心する女性\n"
                "そんな“心の安全基地を求める女性”に刺さりやすい声です。"
            )
        else:
            main_desc = (
                "あなたの安心感のある声は、精神的に疲れやすい男性にとって大きな支えになります。\n"
                "・自信が揺れやすい男性\n"
                "・落ち着いた女性を好む男性\n"
                "・感情を抑えたいタイプの男性\n"
                "そんな“癒しを求める男性”に好まれやすい声です。"
            )

    elif pv > 0.6 and t > 0.5:
        main_title = f"ドラマチックなものが好きな“感情豊かな{target}”"
        if gender == "男性":
            main_desc = (
                "あなたの抑揚のある声は、感情豊かでドラマチックな女性に刺さります。\n"
                "・感受性が強い女性\n"
                "・ストーリー性のある会話が好きな女性\n"
                "・会話にノリと勢いを求める女性\n"
                "そんな“感情で動くタイプの女性”が好む声です。"
            )
        else:
            main_desc = (
                "あなたの抑揚ある声は、感情表現が豊かな男性にとって魅力的に響きます。\n"
                "・アニメ・映画好きな男性\n"
                "・リアクションの多い女性を好む男性\n"
                "・恋愛で“温度感”を求める男性\n"
                "そんな“ドラマチックタイプの男性”に刺さりやすい声です。"
            )

    elif ti < 0.4 and b < 0.5:
        main_title = f"ゆっくり話す“夜型・インドア派の{target}”"
        if gender == "男性":
            main_desc = (
                "ゆったりしたテンポのあなたの声は、夜型・インドアな女性にとても心地よく響きます。\n"
                "・静かな空間が好きな女性\n"
                "・深夜の落ち着いたトーンに弱い女性\n"
                "・じっくり話す時間を大切にする女性\n"
                "そんな“ゆるく過ごしたい女性”に刺さります。"
            )
        else:
            main_desc = (
                "ゆったりした声質は、夜型で落ち着いた男性に相性が良い傾向があります。\n"
                "・静かな趣味を好む男性\n"
                "・低刺激のコミュニケーションを好む男性\n"
                "・温度感の低いトーンを心地よいと感じる男性\n"
                "そんな“インドア派の男性”に好まれやすい声です。"
            )

    else:
        main_title = f"バランスの取れた“聞き上手タイプの{target}”"
        if gender == "男性":
            main_desc = (
                "あなたのバランスの良い声は、過度な刺激が苦手な女性に刺さりやすいです。\n"
                "・話しやすさを重視する女性\n"
                "・落ち着いた男性に惹かれる女性\n"
                "・安心できる距離感を求める女性\n"
                "そんな“聞き上手の女性”と相性が良い声です。"
            )
        else:
            main_desc = (
                "あなたの落ち着いた声は、話しやすい女性に惹かれる男性に刺さります。\n"
                "・相談しやすい女性が好きな男性\n"
                "・柔らかい雰囲気の相手を求める男性\n"
                "・距離感の近すぎない関係を好む男性\n"
                "そんな“聞き上手な男性”と相性の良い声です。"
            )

    # ===== サブタイプ =====
    if e > 0.6 and ti > 0.6:
        sub_title = "短時間でサクッと話したいタイプ"
        if gender == "男性":
            sub_desc = (
                "テンポの良い女性が“あなたとは話しやすい”と感じやすい傾向があります。\n"
                "・効率重視の女性\n"
                "・テンポよく返事をしてほしい女性\n"
                "・短時間の会話で満足できる女性\n"
            )
        else:
            sub_desc = (
                "テンポの良い男性が“あなたと話すと楽しい”と感じやすいです。\n"
                "・サクサク会話したい男性\n"
                "・無駄のない会話を好む男性\n"
                "・テンポで親近感を持つ男性\n"
            )

    elif s > 0.6 and e < 0.6:
        sub_title = "聞き役が好きな・年下タイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの落ち着きは、年下の女性や“頼れる男性”を求める女性に刺さります。\n"
                "・甘えたい女性\n"
                "・優しく包まれたい女性\n"
                "・年上男性が好きな女性\n"
            )
        else:
            sub_desc = (
                "あなたの落ち着いた声は、安心したい男性に魅力的に響きます。\n"
                "・年上女性に惹かれやすい男性\n"
                "・包容力を求める男性\n"
                "・聞き役を求める男性\n"
            )

    elif pv > 0.6 and b > 0.5:
        sub_title = "リアクションが欲しいタイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの明るい声は“反応がほしい女性”に刺さります。\n"
                "・話していて楽しいと思える女性\n"
                "・共感や相づちが欲しい女性\n"
            )
        else:
            sub_desc = (
                "あなたの明るい声は“盛り上がりたい男性”に刺さりやすいです。\n"
                "・リアクションを大事にする男性\n"
                "・おしゃべりが好きな男性\n"
            )

    else:
        sub_title = "ちょうどいい距離感を好むタイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの声は、距離感を大切にしたい女性に刺さります。\n"
                "・近すぎない安心感を求める女性\n"
                "・適度な温度感が好きな女性\n"
            )
        else:
            sub_desc = (
                "あなたの声は、落ち着いた距離感を好む男性に刺さります。\n"
                "・近すぎず遠すぎない関係を好む男性\n"
            )

    # ===== 体格・体質タイプ =====
    if e > 0.7 and ti > 0.6:
        phys_title = f"アクティブで筋肉質寄りの{target}"
        if gender == "男性":
            phys_desc = (
                "スポーツやジムが好きな女性は、テンポの良い声に惹かれやすい傾向があります。\n"
                "・外で動く女性\n"
                "・暑がりな女性\n"
            )
        else:
            phys_desc = (
                "アクティブな男性は、明るくテンポのよい声を好みやすいです。\n"
                "・筋肉質の男性\n"
                "・外出が多い男性\n"
            )

    elif e < 0.4 and s > 0.6 and ti < 0.5:
        phys_title = f"やせ型・華奢でさむがりな{target}"
        if gender == "男性":
            phys_desc = (
                "落ち着いた声は、冷え性寄りの女性や室内派の女性に安心感を与えます。\n"
                "・華奢で寒がりな女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、寒がりで優しい雰囲気の男性によく響きます。\n"
                "・細身の男性\n"
                "・静かな場所が好きな男性\n"
            )

    elif b > 0.6 and e > 0.5 and r < 0.5:
        phys_title = f"健康的でバランスが良い{target}"
        if gender == "男性":
            phys_desc = (
                "ほどよく健康的な女性は、明るく素直な声に好感を持ちやすいです。\n"
                "・規則正しい生活をする女性\n"
            )
        else:
            phys_desc = (
                "健康的な男性は、明るく素直な声を魅力的に感じます。\n"
                "・よく歩く男性\n"
            )

    elif r > 0.6 and e > 0.5:
        phys_title = f"アウトドア寄りの“暑がりな{target}”"
        if gender == "男性":
            phys_desc = (
                "アウトドアが好きな女性は、少しザラつく男らしい声に惹かれます。\n"
                "・汗っかきの女性\n"
            )
        else:
            phys_desc = (
                "アウトドア好きな男性は、個性ある声に弱い傾向があります。\n"
                "・フェス好き男性\n"
            )

    elif ti < 0.4 and b < 0.5 and e < 0.6:
        phys_title = f"室内好き・冷え性寄りの{target}"
        if gender == "男性":
            phys_desc = (
                "インドア寄りの女性は、ゆっくりしたテンポの声に心を許しやすいです。\n"
                "・家時間を好む女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、室内でゆったり過ごす男性に好まれます。\n"
                "・落ち着いた男性\n"
            )

    else:
        phys_title = f"雰囲気重視の{target}"
        if gender == "男性":
            phys_desc = (
                "あなたの声は体型より“雰囲気”を重視する女性に響きます。\n"
                "・声や空気感で人を見る女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、トータルの雰囲気を大事にする男性に刺さります。\n"
                "・気配りを感じたい男性\n"
            )

    # ===== 恋愛シチュ =====
    if e > 0.7 and b > 0.6:
        situ_title = "相手がちょっと酔って甘えモードの瞬間"
        if gender == "男性":
            situ_desc = (
                "明るい声は、女性が少し酔って甘えたときに強く刺さります。\n"
                "・帰り道の軽いツッコミ\n"
                "・ふざけ合いの時の一言\n"
            )
        else:
            situ_desc = (
                "男性が少し気が緩んだ瞬間、あなたの声が“可愛い”と感じられやすいです。\n"
                "・酔いが回った男性への軽い冗談\n"
            )

    elif s > 0.7 and e < 0.6:
        situ_title = "落ち込んだ相手にそっとかける一言"
        if gender == "男性":
            situ_desc = (
                "女性が弱っているとき、あなたの落ち着いた一言が深く刺さります。\n"
                "・「今日は頑張ったね」\n"
                "・「無事帰れそう？」\n"
            )
        else:
            situ_desc = (
                "男性の心が弱った瞬間、あなたの声が特に心に沁みます。\n"
                "・「大丈夫？」\n"
                "・「今日は疲れたね」\n"
            )

    elif pv > 0.6 and t > 0.5:
        situ_title = "ちょっと弱気な“お願いごと”が刺さる瞬間"
        if gender == "男性":
            situ_desc = (
                "女性はギャップに弱い傾向があり、弱気な声が刺さりやすいです。\n"
                "・「もうちょっと一緒にいたい」\n"
            )
        else:
            situ_desc = (
                "男性は感情の揺れを感じた瞬間に惹かれやすいです。\n"
                "・「手つないでもいい？」\n"
            )

    elif ti < 0.4 and b < 0.5:
        situ_title = "寝落ち寸前のやり取り"
        if gender == "男性":
            situ_desc = (
                "眠たげな女性に対して、あなたの落ち着いた声がとても刺さります。\n"
                "・深夜の「大丈夫？」\n"
            )
        else:
            situ_desc = (
                "男性が眠くなっている瞬間、あなたの声の柔らかさが沁みます。\n"
                "・「無理しないでね」\n"
            )

    elif r > 0.6 and e > 0.5:
        situ_title = "距離が近いシーンでの落ち着いた声"
        if gender == "男性":
            situ_desc = (
                "耳元で話すあなたの低めの声は、女性に刺さりやすいです。\n"
                "・映画館や車内の小声\n"
            )
        else:
            situ_desc = (
                "男性は距離が近い瞬間、あなたの落ち着いた声にドキッとしやすいです。\n"
                "・「ちょっと来て」\n"
            )

    else:
        situ_title = "日常の“自然なワンシーン”"
        if gender == "男性":
            situ_desc = (
                "あなたの声は、女性の日常の中でじんわり刺さります。\n"
                "・「それいいね」などの自然な一言\n"
            )
        else:
            situ_desc = (
                "男性は日常的な優しい声かけで恋に落ちることが多いです。\n"
                "・歩きながらの雑談\n"
            )

    return {
        "main_title": main_title,
        "main_desc": main_desc,
        "sub_title": sub_title,
        "sub_desc": sub_desc,
        "phys_title": phys_title,
        "phys_desc": phys_desc,
        "situ_title": situ_title,
        "situ_desc": situ_desc,
    }




# ========= 声の特徴レポート =========
def explain_traits(traits: dict) -> str:
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    lines = []
    lines.append("● あなたの声のざっくり性質（0〜1）")
    lines.append(f"- 元気さ（energy）      : {e:.2f}")
    lines.append(f"- 緊張感（tension）     : {t:.2f}")
    lines.append(f"- 安定感（stability）   : {s:.2f}")
    lines.append(f"- 明るさ（brightness）  : {b:.2f}")
    lines.append(f"- ザラつき（roughness） : {r:.2f}")
    lines.append(f"- 抑揚（pitch_var）      : {pv:.2f}")
    lines.append(f"- テンポ（tempo_index） : {ti:.2f}")
    lines.append("")

    lines.append("● 声の特徴から見える“印象の方向性”")
    if e > 0.7:
        lines.append("・エネルギーが高めで、前向き・アクティブな印象を与えやすい声です。")
    elif e < 0.3:
        lines.append("・エネルギー控えめで、落ち着き・ゆったり感を感じさせる声です。")

    if s > 0.7:
        lines.append("・安定感が高く、聞き手に“安心感”を与えやすいタイプです。")
    elif s < 0.3:
        lines.append("・声の揺れがやや大きめで、感情や迷いがにじみやすい声です。")

    if b > 0.7:
        lines.append("・トーンが明るめで、距離を縮めやすいフレンドリーな声質です。")
    elif b < 0.3:
        lines.append("・トーンが落ち着き寄りで、しっとり・大人っぽい印象になりやすいです。")

    if pv > 0.7:
        lines.append("・抑揚が大きく、感情表現豊かな印象を与えやすいです。")
    elif pv < 0.3:
        lines.append("・抑揚控えめで、“穏やか・クール”寄りの印象を持たれやすいです。")

    if len(lines) == 2:
        lines.append("・大きな偏りはなく、場面に合わせて印象を変えやすいバランス型の声です。")

    lines.append("")
    lines.append("※この診断は、声の特徴から“好きになりやすそうな人のタイプ”をそれっぽく遊ぶエンタメ用です。")
    lines.append("　実際の人間関係とは必ずしも一致しないので、ネタとして楽しんでください。")

    return "\n".join(lines)


# ========= Streamlit UI =========
def main():
    st.set_page_config(page_title="あなたの声がささる人", page_icon="🎧", layout="centered")

    st.title("🎧 あなたの声が刺さる人")
    st.caption(
        "声のテンション・安定感・明るさなどから、"
        "“どんな人があなたの声にハマりやすいか”をざっくり診断するエンタメツールです。"
    )

    st.subheader("性別を選択してください")
    gender = st.radio("あなたの性別は？", ["男性", "女性"], horizontal=True)

    st.markdown("---")
    st.subheader("1. 声の録音")

    st.write(
        "普段どおりの話し方で、30秒〜1分ほど話した音声がおすすめです。\n"
        "最近あったことや、好きなものについて話して録音すると、結果がそれっぽくなります。"
    )

    uploaded = st.audio_input("🎤 マイクで録音してください（30秒〜1分）")

    if uploaded is None:
        st.info("マイクで録音すると診断ボタンが表示されます。")
        return

    # 一時ファイルに保存
    audio_bytes = uploaded.getbuffer()
    suffix = ".wav"  # audio_input は WAV 形式
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(tmp_path)

    if st.button("あなたの声がハマる人を診断する"):
        with st.spinner("声の雰囲気から“あなたの声が刺さりやすい人”を分析中..."):
            try:
                features = extract_voice_features(tmp_path)
                traits = score_voice_traits(features)
                fan_info = classify_fan_types(traits, gender)
                trait_text = explain_traits(traits)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        st.markdown("---")
        st.subheader("2. メインであなたの声を好きになりやすい人")

        st.markdown(f"### 💘 {fan_info['main_title']}")
        st.write(fan_info["main_desc"])

        st.markdown("---")
        st.subheader("3. サブで刺さりやすい人のタイプ")

        st.markdown(f"### ✨ {fan_info['sub_title']}")
        st.write(fan_info["sub_desc"])

        st.markdown("---")
        st.subheader("4. 別の視点：こういう見た目・体質の人にも刺さりやすい")

        st.markdown(f"### 🧍 {fan_info['phys_title']}")
        st.write(fan_info["phys_desc"])

        st.markdown("---")
        st.subheader("5. 恋愛で“あなたの声”が最も刺さるシチュエーション")

        st.markdown(f"### ❤️ {fan_info['situ_title']}")
        st.write(fan_info["situ_desc"])

        st.markdown("---")
        st.subheader("6. あなたの声の特徴レポート")
        st.text(trait_text)

        st.markdown("---")
        st.caption(
            "※この診断はジョーク用です。実際の相性や恋愛・人間関係を保証するものではありません。\n"
            "　“自分の声がどんな人・どんな場面にハマりそうか”を、ゆるく楽しむツールとして使ってください。"
        )


if __name__ == "__main__":
    main()