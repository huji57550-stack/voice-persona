# voice_fan_type_full.py
# 🎧 あなたの声がささる人

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

# ========= 音声特徴量の抽出 (修正) =========
def extract_voice_features(file_path: str) -> dict:
    """音声ファイルから簡易特徴量を抽出し、基本的なデータチェックを行う"""
    y, sr = librosa.load(file_path, sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    
    # 【変更点1: 長さのチェック】
    if duration < 10.0:
        raise ValueError("Audio duration is too short (less than 10 seconds).")

    # 【変更点2: 無音チェック】
    # 簡易的にピーク音量をチェックし、非常に小さい場合は無音と判断
    peak_amplitude = np.max(np.abs(y))
    if peak_amplitude < 0.005:  # 閾値を調整可能 (例: 0.005はかなり小さい音)
        raise ValueError("Audio is too quiet or appears to be silent.")

    # 無音対策（小さなノイズを加える処理は不要になるが、念のため残す）
    if peak_amplitude < 1e-4:
        y = y + np.random.normal(0, 1e-4, size=len(y))

    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 【改善点1: MFCCs】音色・声質の詳細
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc1_mean = float(np.mean(mfccs[1, :])) # MFCC 1の平均 (声のトーンに大きく影響)

    # 【改善点2: デルタ特徴量】音量の時間的変化率
    # デルタRMS (音量の変化の速さ)
    rms_delta = librosa.feature.delta(rms)
    rms_delta_mean = float(np.mean(np.abs(rms_delta))) # 絶対値平均で変動の大きさを計測

    # ピッチ（声の高さ）
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_valid = f0[np.isfinite(f0)]
    if len(f0_valid) == 0:
        f0_mean = 0.0
        f0_std = 0.0
        f0_range = 0.0 # 【改善点3: ピッチレンジ】
    else:
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        f0_range = float(np.max(f0_valid) - np.min(f0_valid)) # 【改善点3: ピッチレンジ】

    return {
        "duration": float(duration),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_delta_mean": rms_delta_mean, # 【改善点2】
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "tempo": float(tempo),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range, # 【改善点3】
        "mfcc1_mean": mfcc1_mean, # 【改善点1】
    }


# ========= スコアリング (修正) =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def score_voice_traits(features: dict) -> dict:
    rms_mean = features["rms_mean"]
    rms_std = features["rms_std"]
    rms_delta_mean = features["rms_delta_mean"] # 【改善点2】
    zcr_mean = features["zcr_mean"]
    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_std = features["f0_std"]
    f0_range = features["f0_range"] # 【改善点3】
    mfcc1_mean = features["mfcc1_mean"] # 【改善点1】

    # 元気さ (エネルギー)
    # テンポと平均音量に、音量の変動の大きさも追加 (より活動的な声を評価)
    # 【改善点: rms_delta_meanを組み込み】
    energy = (
        0.5 * normalize(rms_mean, 0.005, 0.05) 
        + 0.3 * normalize(tempo, 60, 180)
        + 0.2 * normalize(rms_delta_mean, 0.0001, 0.005) 
    )
    
    # 緊張感
    # 【改善点: 緊張感の指標としてmfcc1_meanを組み込み (高周波成分)】
    tension = (
        0.3 * normalize(zcr_mean, 0.01, 0.2)
        + 0.2 * normalize(rms_std, 0.0, 0.03)
        + 0.3 * normalize(centroid_mean, 1000, 4000)
        + 0.2 * normalize(mfcc1_mean, -20.0, 30.0) # MFCC1がプラス寄りだと声が高く硬い印象
    )
    
    # 安定感
    # 【改善点: f0_rangeを組み込み (抑揚の幅が狭いほど安定)】
    stability = (
        1.0 
        - 0.4 * normalize(f0_std, 0, 40)
        - 0.3 * normalize(rms_std, 0.0, 0.03)
        - 0.3 * normalize(f0_range, 0, 150) # ピッチの幅も安定感に寄与
    )
    stability = float(np.clip(stability, 0.0, 1.0))
    
    # 明るさ
    brightness = normalize(centroid_mean, 1000, 4000)
    
    # ザラつき
    roughness = normalize(zcr_mean, 0.01, 0.2)
    
    # 抑揚
    # 【改善点: f0_stdとf0_rangeの両方で抑揚を評価】
    pitch_var = 0.5 * normalize(f0_std, 0, 40) + 0.5 * normalize(f0_range, 0, 150)
    
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


ご提示いただいた、classify_fan_types 関数のロジック（メインタイプ、サブタイプ、体格・体質、恋愛シチュエーション）について、分岐を細分化し、記述をより詳細に修正しました。

特に、ビッグ・ファイブ特性(ext, agr, con, sta, ope) を活用して、各タイプの説明をより複合的で具体的にしています。

修正版コード: classify_fan_types 関数 🛠️
Python

# ========= 「あなたの声が好きな人」タイプ判定（＋体格＋恋愛シチュ） (最終修正) =========
def classify_fan_types(traits: dict, big_five_scores: dict, gender: str) -> dict:
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]
    
    # Big Five スコアを取得
    ext = big_five_scores["Extraversion"]       # 外向性
    agr = big_five_scores["Agreeableness"]      # 協調性
    con = big_five_scores["Conscientiousness"]  # 誠実性
    sta = big_five_scores["Emotional_Stability"]# 情緒安定性
    ope = big_five_scores["Openness"]           # 開放性

    # 男女別ターゲットの呼び方
    if gender == "男性":
        target = "女性"
    else:
        target = "男性"


    # ===== メインタイプ (Big Five特性を最大限に活用) =====
    
    # 1. テンションを上げたい“陽キャ寄り”のターゲット（元気に加え、外交性が高い）
    if e > 0.7 and b > 0.6 and pv > 0.6 and ext > 0.7:
        main_title = f"🚀 テンションを上げたい“陽キャ寄り”の{target} (刺激・活動性重視)"
        if gender == "男性":
            main_desc = (
                "あなたの明るく抑揚のある声は、**外向的でアクティブ**な女性に強く響きます。場の空気を読むより、盛り上がりを重視するタイプです。\n"
                "・ノリが良く、明るい会話を好む女性\n"
                "・社交的な場で一緒に楽しめる男性に惹かれる女性\n"
                "・元気の素となる存在を常に求めている女性\n"
                f"そんな“**高外向性・活動性重視の{target}**”があなたの声にハマりやすい傾向があります。"
            )
        else:
            main_desc = (
                "あなたの明るく抑揚のある声は、**気分を高揚させたい外向的な男性**に刺さりやすいです。会話の勢いとリアクションが重要です。\n"
                "・ユーモアや楽しい雰囲気の女性が好きな男性\n"
                "・話していてエネルギーをもらえる相手を求める男性\n"
                "・一緒に新しいことに挑戦したがる男性\n"
                f"そんな“**高外向性・快活な{target}**”に好まれやすいタイプです。"
            )

    # 2. せかされるのが苦手な“おだやか系”のターゲット（低エネ・高安定/協調）
    elif e < 0.4 and s > 0.6 and b < 0.5 and sta > 0.6 and agr > 0.6:
        main_title = f"🧘 せかされるのが苦手な“おだやか系”の{target} (低刺激・協調性重視)"
        if gender == "男性":
            main_desc = (
                "あなたの落ち着いた安定した声は、**刺激を避け、調和を求める協調性の高い女性**に深い安心感を与えます。\n"
                "・自分のペースや時間を乱されたくない女性\n"
                "・感情的にならず、穏やかな男性の声に癒される女性\n"
                "・周囲の環境に敏感で、静かな安心感を好む女性\n"
                f"そんな“**高協調性・低刺激重視の{target}**”に刺さりやすい声です。"
            )
        else:
            main_desc = (
                "あなたの落ち着いた声は、**穏やかで協調的**な性格の男性に心地よく響きます。過度なプレッシャーを嫌います。\n"
                "・趣味や日常を静かに楽しみたい男性\n"
                "・争いを避け、優しく話してくれる相手に惹かれる男性\n"
                "・安心感と癒しをコミュニケーションに求める男性\n"
                f"そんな“**高協調性・おだやかさ重視の{target}**”があなたの声を好みやすい傾向があります。"
            )

    # 3. 安心感を求める“ちょっと不安気味なターゲット”（高安定・低緊張・高誠実）
    elif s > 0.7 and t < 0.5 and sta > 0.7 and con > 0.6:
        main_title = f"🛡️ 心の安全基地を求める“不安・緊張を抱えやすい{target}” (高誠実・情緒安定重視)"
        if gender == "男性":
            main_desc = (
                "あなたの安定した声は、**緊張や不安を感じやすい女性**にとって「**心の安全基地**」になりやすいです。あなたの誠実性が信頼感を生みます。\n"
                "・ストレスや緊張を感じやすく、心の支えを求める女性\n"
                "・ブレない、信頼できる男性の声に安心感を覚える女性\n"
                "・誠実で真面目な対応を重視する女性\n"
                f"そんな“**高誠実性・安心感を求める{target}**”に刺さりやすい声です。"
            )
        else:
            main_desc = (
                "あなたの安心感のある声は、**精神的な安定や自信の揺らぎ**を感じやすい男性にとって、大きな支えになります。\n"
                "・感情の波を抑えたい、または自信を回復したい男性\n"
                "・地に足の着いた、誠実な女性を好む男性\n"
                "・落ち着いたトーンで物事を解決してくれる相手を求める男性\n"
                f"そんな“**癒しと信頼性を求める{target}**”に好まれやすい声です。"
            )

    # 4. ドラマチックなものが好きな“感情豊かなターゲット”（高抑揚・高緊張・高開放）
    elif pv > 0.6 and t > 0.5 and ope > 0.6:
        main_title = f"🎭 感動と刺激を求める“感情豊かな{target}” (高開放性・表現力重視)"
        if gender == "男性":
            main_desc = (
                "あなたの抑揚と緊張感のある声は、**新しい経験やドラマを愛する開放性の高い女性**に強く刺さります。\n"
                "・感受性が強く、感情の動きを大切にする女性\n"
                "・会話の「熱量」や「ストーリー性」を重視する女性\n"
                "・変化や刺激を求め、単調な会話を好まない女性\n"
                f"そんな“**高開放性・ドラマチックな{target}**”が好む声です。"
            )
        else:
            main_desc = (
                "あなたの抑揚ある声は、**芸術や感情表現に価値を置く開放的な男性**にとって魅力的に響きます。\n"
                "・趣味や文化的な話題で盛り上がれる女性を好む男性\n"
                "・恋愛において高い「温度感」と「情熱」を求める男性\n"
                "・話すことで感情を共有できる相手を求める男性\n"
                f"そんな“**高開放性・感情表現を重視する{target}**”に刺さりやすい声です。"
            )

    # 5. ゆっくり話す“夜型・インドア派のターゲット”（低テンポ・低明るさ・協調性中庸）
    elif ti < 0.4 and b < 0.5 and agr > 0.4 and agr < 0.7:
        main_title = f"🌙 ゆったりとした時間を好む“夜型・インドア派の{target}” (低テンポ・中庸な距離感)"
        if gender == "男性":
            main_desc = (
                "ゆったりしたテンポのあなたの声は、**家時間や夜の落ち着きを好む内向的な女性**にとても心地よく響きます。\n"
                "・静かでパーソナルな空間を大切にする女性\n"
                "・深夜のラジオのような、低刺激で落ち着いたトーンを好む女性\n"
                "・人間関係で程よい距離感を重視する女性\n"
                f"そんな“**低刺激・ゆったりペースの{target}**”に刺さります。"
            )
        else:
            main_desc = (
                "ゆったりした声質は、**インドアでマイペースに過ごす男性**に相性が良い傾向があります。\n"
                "・派手な刺激や大人数を避け、落ち着いた趣味を好む男性\n"
                "・低温度で、長時間じっくりとした会話ができる相手を好む男性\n"
                "・心地よい静寂と、会話のペースを重視する男性\n"
                f"そんな“**低テンポ・マイペースな{target}**”に好まれやすい声です。"
            )

    else:
        main_title = f"⚖️ バランスの取れた“柔軟で聞き上手な{target}” (全方位型)"
        if gender == "男性":
            main_desc = (
                "あなたの声は大きな偏りがなく、**どんなタイプの人とも話しやすい柔軟性**を持っています。特に過度な刺激を苦手とする女性に心地よく受け入れられます。\n"
                "・押しが強い会話が苦手で、話しやすさを重視する女性\n"
                "・TPOに合わせて会話のトーンを変えられる男性に安心感を覚える女性\n"
                "・自分の話を優しく聞いてくれる男性に惹かれる女性\n"
                f"そんな“**柔軟性と聞く力を重視する{target}**”と相性が良い声です。"
            )
        else:
            main_desc = (
                "あなたの落ち着いた声は、**コミュニケーションの安定性**を重視する男性に刺さります。相手のペースを尊重する「聞き上手」な印象を与えます。\n"
                "・相談しやすく、安心感のある女性が好きな男性\n"
                "・感情的な起伏が少なく、柔らかい雰囲気の相手を求める男性\n"
                "・適切な距離感を保ち、長く付き合える関係を好む男性\n"
                f"そんな“**バランス重視の{target}**”と相性の良い声です。"
            )


    # ---
    # ===== サブタイプ (会話のテンポ・関わり方) =====
    # ---

    if e > 0.6 and ti > 0.6 and con > 0.6: # 高エネルギー・高テンポ・高誠実
        sub_title = "テキパキと結論を出したい効率重視タイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの声は、**効率性と誠実性**を重視する女性にとって「話が早い」「信頼できる」と感じやすいです。\n"
                "・仕事やタスクをテキパキこなしたい女性\n"
                "・テンポよく質問と返答をしてほしい女性\n"
                "・短時間の濃密な会話で満足できる女性\n"
            )
        else:
            sub_desc = (
                "あなたの声は、**無駄を嫌い、サクサクと会話を進めたい男性**に心地よいテンポを提供します。\n"
                "・結論から話すことを好む男性\n"
                "・知的で合理的なコミュニケーションを好む男性\n"
            )

    elif s > 0.6 and e < 0.6 and agr > 0.7: # 高安定・低エネルギー・高協調
        sub_title = "甘えたい・包容力を求める・年下タイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの落ち着きと安定感は、**心の支えや包容力**を求める女性に深く刺さります。\n"
                "・年下の女性や、安心感を求めて甘えたい女性\n"
                "・自分の話を遮らず、優しく包み込んでくれる男性を求める女性\n"
            )
        else:
            sub_desc = (
                "あなたの声は、**精神的な安らぎや母親的な安心感**を求める男性に魅力的に響きます。\n"
                "・年上女性に惹かれやすい、または包容力を求める男性\n"
                "・自分が主導権を握るより、聞き役を求める男性\n"
            )

    elif pv > 0.6 and b > 0.5 and ext > 0.6: # 高抑揚・高明るさ・高外向
        sub_title = "共感と大きなリアクションが欲しいムードメーカータイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの明るく抑揚のある声は、**会話の楽しさや反応**を重視する女性に刺さります。\n"
                "・話していて感情を豊かに表現してくれる男性を好む女性\n"
                "・自分の話題に共感や大きな相づちが欲しい女性\n"
            )
        else:
            sub_desc = (
                "あなたの明るい声は、**場のムードを盛り上げたい外向的な男性**に刺さりやすいです。\n"
                "・会話で熱量を共有したい男性\n"
                "・リアクションの多い、おしゃべり好きな女性を好む男性\n"
            )

    else:
        sub_title = "心理的な安全性を重視するタイプ"
        if gender == "男性":
            sub_desc = (
                "あなたの声は、**人間関係で心理的な安全性**と距離感を大切にしたい女性に刺さります。\n"
                "・急激な関係性の変化を望まない女性\n"
                "・過度な情熱や、一方的な熱量を求めない女性\n"
            )
        else:
            sub_desc = (
                "あなたの声は、**落ち着いた大人の付き合い**を好む男性に響きます。\n"
                "・近すぎず遠すぎない、お互いを尊重しあえる関係を好む男性\n"
            )

    # ---
    # ===== 体格・体質タイプ (声の物理特性に基づく) =====
    # ---

    if e > 0.7 and ti > 0.7 and ext > 0.6: # 高エネルギー・高テンポ・高外向
        phys_title = f"💪 アクティブで筋肉質寄りの{target} (健康・エネルギッシュ)"
        if gender == "男性":
            phys_desc = (
                "スポーツやジム、アウトドアなど**活動的な趣味を持つ女性**は、あなたのエネルギッシュな声に惹かれやすい傾向があります。\n"
                "・健康的な体型で、外で動くことを好む女性\n"
                "・基礎代謝が高く、寒がりではない女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、**活動的でフィジカルな強さ**を持つ男性に好まれます。\n"
                "・筋力トレーニングやスポーツを好む男性\n"
                "・体力があり、外出や旅行を好む男性\n"
            )

    elif e < 0.4 and s > 0.6 and ti < 0.5 and sta > 0.6: # 低エネルギー・高安定・低テンポ・高情緒安定
        phys_title = f"📚 やせ型・華奢でさむがりな{target} (インドア・繊細)"
        if gender == "男性":
            phys_desc = (
                "あなたの落ち着いた声は、**内向的で体質的に繊細な女性**に深い安心感を与えます。\n"
                "・冷え性や華奢な体型で、室内での活動を好む女性\n"
                "・刺激の強い環境を避けたい女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、**穏やかで室内での趣味**を好む男性によく響きます。\n"
                "・細身で、落ち着いた場所での会話を好む男性\n"
                "・体温調整が苦手で、心地よい温度を求める男性\n"
            )

    elif b > 0.6 and e > 0.5 and r < 0.5 and con > 0.6: # 高明るさ・中高エネ・低ザラつき・高誠実
        phys_title = f"🍎 健康的でバランスが良い{target} (規則正しい生活)"
        if gender == "男性":
            phys_desc = (
                "あなたの明るく素直な声は、**規則正しく健康的な生活**を送る女性に好感を持ちやすいです。\n"
                "・心身の健康を意識し、調和の取れた状態を好む女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、**全体的にバランスが取れた健康志向の男性**を魅力的に感じさせます。\n"
                "・節度のある生活を送り、急激な変化を好まない男性\n"
            )

    elif r > 0.6 and e > 0.5 and pv > 0.5: # 高ザラつき・中高エネ・中高抑揚
        phys_title = f"⛰️ アウトドア寄りの“暑がりで個性的”な{target}"
        if gender == "男性":
            phys_desc = (
                "あなたの低く少しザラつく声は、**個性的で野性的な魅力**に惹かれる女性に刺さります。\n"
                "・外遊びやイベント好きで、情熱的な気質を持つ女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、**個性や情熱的な表現**を好む男性に強く響きます。\n"
                "・ワイルドな魅力や、自己主張の強い声に惹かれる男性\n"
            )

    else:
        phys_title = f"✨ 雰囲気重視の{target} (体格よりもオーラ重視)"
        if gender == "男性":
            phys_desc = (
                "あなたの声は、体型や体質といった物理的な要素よりも、**相手の醸し出す声の雰囲気やオーラ**を重視する女性に響きます。\n"
                "・声のトーンや話し方から人間性やムードを読み取る女性\n"
            )
        else:
            phys_desc = (
                "あなたの声は、**ディテールよりも全体的な空気感**を大事にする男性に刺さります。\n"
                "・声から感じる優しさや気配りを重視する男性\n"
            )

    # ---
    # ===== 恋愛シチュ (最も声が魅力的になる瞬間) =====
    # ---

    if e > 0.7 and b > 0.6 and ext > 0.7: # 高エネ・高明るさ・高外向
        situ_title = "🍻 相手が少し酔って甘えモードの瞬間 (陽気なツッコミ)"
        if gender == "男性":
            situ_desc = (
                "女性が**心を開いて甘えや冗談を求めてくる瞬間**に、あなたの明るい声の陽気さが強く刺さります。\n"
                "・二人でいる時の、元気で親愛の込められた軽いツッコミ\n"
                "・楽しいムードの中での、ふざけ合いの一言\n"
            )
        else:
            situ_desc = (
                "男性が**気が緩み、心に素直になった瞬間**、あなたの声が特に“可愛い”“魅力的”と感じられやすいです。\n"
                "・酔いが回った男性への、安心感のある優しい冗談\n"
            )

    elif s > 0.7 and e < 0.6 and sta > 0.7: # 高安定・低エネ・高情緒安定
        situ_title = "☔ 落ち込んだ相手にそっとかける安定の一言"
        if gender == "男性":
            situ_desc = (
                "女性が**不安やストレスで弱っているとき**、あなたの落ち着いた声は「絶対的な安全」を与えます。あなたの安定感が真価を発揮します。\n"
                "・「焦らなくて大丈夫だよ」\n"
                "・「俺がいるから安心して」\n"
            )
        else:
            situ_desc = (
                "男性が**心が疲弊し、支えを求めている瞬間**、あなたの声が特に心に沁みます。\n"
                "・「頑張りすぎなくていいよ」\n"
                "・「今日は疲れたね」といった共感の言葉\n"
            )

    elif pv > 0.6 and t > 0.5 and ope > 0.6: # 高抑揚・高緊張・高開放性
        situ_title = "💖 感情の揺れを見せる“お願いごと”が刺さる瞬間 (ギャップ)"
        if gender == "男性":
            situ_desc = (
                "普段明るいあなたが**少しだけ弱気な感情を見せたとき**、そのギャップが女性の心を掴みます。抑揚が感情の深さを伝えます。\n"
                "・真剣なトーンで「もうちょっと一緒にいたい」\n"
            )
        else:
            situ_desc = (
                "男性は**あなたの感情の揺れ**を感じた瞬間に、ドラマチックな惹かれ方をしやすいです。\n"
                "・少し緊張しながら「手つないでもいい？」\n"
            )

    elif ti < 0.4 and b < 0.5 and agr > 0.5: # 低テンポ・低明るさ・中協調
        situ_title = "😴 寝落ち寸前のゆったりとした時間"
        if gender == "男性":
            situ_desc = (
                "女性が**眠たげで、警戒心が解けている時**、あなたのゆっくりとした優しい声が、最高の心地よさを提供します。\n"
                "・深夜の「無理しないで、おやすみ」\n"
            )
        else:
            situ_desc = (
                "男性が**一日の疲れでリラックスしている瞬間**、あなたの声の柔らかさとテンポが深く沁みます。\n"
                "・「無理しないでね」といった心身を気遣う一言\n"
            )

    elif r > 0.6 and e > 0.5 and s > 0.5: # 高ザラつき・中高エネ・中高安定
        situ_title = "👂 距離が近いシーンでの落ち着いた囁き"
        if gender == "男性":
            situ_desc = (
                "あなたの**低く少しザラつく落ち着いた声**は、物理的に距離が近い瞬間、女性に強いドキドキ感を与えます。\n"
                "・映画館や車内での耳元での小声の会話\n"
            )
        else:
            situ_desc = (
                "男性は、**予期せぬ接近**があった瞬間、あなたの落ち着いた声にドキッとしやすいです。\n"
                "・「ちょっと来て」といった、さりげなくリードする一言\n"
            )

    else:
        situ_title = "🌱 日常の“自然なワンシーン”"
        if gender == "男性":
            situ_desc = (
                "あなたの声は、**過剰な演出がない日常の自然な瞬間**に、女性の心にじんわりと染み渡ります。\n"
                "・「それいいね」「そうだね」などの、会話を肯定する自然な一言\n"
            )
        else:
            situ_desc = (
                "男性は、あなたの**日常的な優しさや気遣い**を声から感じ取り、恋に落ちることが多いです。\n"
                "・並んで歩きながらの、何気ない雑談\n"
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

# ========= 声の特徴レポート (修正・記述詳細化) =========
# 【変更点1: 引数に big_five_scores を追加】
def explain_traits(traits: dict, big_five_scores: dict) -> str:
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    # Big Five スコアを取得
    ext = big_five_scores["Extraversion"]       # 外向性
    agr = big_five_scores["Agreeableness"]      # 協調性
    con = big_five_scores["Conscientiousness"]  # 誠実性
    sta = big_five_scores["Emotional_Stability"]# 情緒安定性
    ope = big_five_scores["Openness"]           # 開放性

    lines = []
    lines.append("● あなたの声のざっくり性質（0〜1）")
    # ... (既存の 7 つの特性表示は変更なし) ...
    lines.append(f"- 元気さ（energy）      : {e:.2f}")
    lines.append(f"- 緊張感（tension）     : {t:.2f}")
    lines.append(f"- 安定感（stability）   : {s:.2f}")
    lines.append(f"- 明るさ（brightness）  : {b:.2f}")
    lines.append(f"- ザラつき（roughness） : {r:.2f}")
    lines.append(f"- 抑揚（pitch_var）      : {pv:.2f}")
    lines.append(f"- テンポ（tempo_index） : {ti:.2f}")
    lines.append("")
    
    # Big Five スコアの表示を挿入
    lines.append("● 声の印象から推定されるビッグ・ファイブ特性（0〜1）")
    lines.append(f"- 外向性（Extraversion）     : {ext:.2f}")
    lines.append(f"- 協調性（Agreeableness）   : {agr:.2f}")
    lines.append(f"- 誠実性（Conscientiousness）: {con:.2f}")
    lines.append(f"- 情緒安定性（Emotional_Stability）: {sta:.2f}")
    lines.append(f"- 開放性（Openness）         : {ope:.2f}")
    lines.append("")

    lines.append("● 声の特徴から見える“印象の方向性”（詳細版）")
    
    # --- エネルギー/外向性 ---
    if e > 0.8 and ext > 0.8:
        lines.append("・**【陽気・社交的】** エネルギーと外向性が非常に高く、場の雰囲気を明るくする**リーダー的な印象**を与えやすいです。")
    elif e > 0.7 and ext > 0.6:
        lines.append("・**【活動的】** テンポ、音量、抑揚があり、話すことを楽しんでいる**快活な印象**です。多くの人が魅力を感じやすいタイプです。")
    elif e < 0.3 and ext < 0.3:
        lines.append("・**【内省的・控えめ】** エネルギーと外向性が低く、**落ち着きがあり、じっくり話を聞く人**という印象です。穏やかな関係を好む人に響きます。")
    elif e < 0.5 and ext > 0.6:
        lines.append("・**【親しみやすい】** 外向性は高いものの、エネルギーは中程度で、**話しやすい友達のような親近感**を与える傾向があります。")

    # --- 安定感/情緒安定性/誠実性 ---
    if s > 0.8 and sta > 0.8:
        lines.append("・**【不動の安定感】** 声の揺れが少なく情緒安定性が極めて高いため、**頼れる・ブレない存在**として強い安心感を抱かせる声です。")
    elif s > 0.7 and con > 0.7:
        lines.append("・**【信頼性・計画性】** 安定感と誠実性が高く、**論理的で信頼できる話し方**です。重要な場面での説得力が増しやすいタイプです。")
    elif s < 0.3 and sta < 0.4:
        lines.append("・**【繊細・感情的】** 声の揺れが大きく、感情の起伏や緊張が声に出やすいです。**感受性が豊かな人**という印象につながります。")
    
    # --- 抑揚/開放性 ---
    if pv > 0.7 and ope > 0.7:
        lines.append("・**【表現豊か・個性的】** 抑揚が大きく開放性も高いため、**話の内容に独自性があり、聞き手を飽きさせない**魅力的な声です。")
    elif pv < 0.3 and ope < 0.4:
        lines.append("・**【クール・実用的】** 抑揚が少なく開放性も低い場合、**シンプルで実用的なコミュニケーション**を好む、クールな印象を持たれやすいです。")

    # --- 明るさ/緊張感/協調性 ---
    if b > 0.7 and t < 0.5 and agr > 0.7:
        lines.append("・**【優しさ・包容力】** 明るいトーンでありながら緊張感が低く協調性が高い場合、**誰に対しても優しく、受け入れる**包容力のある印象です。")
    elif t > 0.6 and r > 0.5:
        lines.append("・**【カリスマ性/緊張感】** ザラつきと緊張感が高めの場合、**強い意思や個性を感じさせる**声です。独特のカリスマ性につながることもあります。")
    elif ti < 0.4 and b < 0.5:
        lines.append("・**【癒やし・安らぎ】** テンポが遅くトーンが落ち着いている場合、聞いている人に**リラックス効果**を与える「癒やし声」の性質があります。")

    
    if len(lines) <= 2: # 上記の条件にほとんど引っかからない場合
        lines.append("・**【バランス型】** 大きな偏りがなく、特定の印象に固定されない**柔軟性の高い声**です。場面や相手に合わせて印象を変えやすいでしょう。")

    lines.append("")
    lines.append("※この診断は、声の特徴と心理学的傾向を関連付けたエンタメ用です。")
    lines.append("　実際の人間関係とは必ずしも一致しないので、あくまでネタとして楽しんでください。")

    return "\n".join(lines)


# ========= ビッグ・ファイブ性格特性のスコアリング (修正) =========
def score_big_five_traits(traits: dict) -> dict:
    """
    声の特性スコア (0-1) を Big Five 性格特性 (0-1) に変換する。
    ご提示のロジック (抑揚、安定性、テンポなど) に基づいて再設計。
    ※ 既存の traits は 0-1 スケールのため、Zスコアのロジックを近似で組み込む。
    """
    # 既存の traits スコアを定義
    # (z[1]〜z[7] に対応する特性を近似的に使用)
    pitch_var = traits["pitch_var"]      # 抑揚 (近似 z[1] / z[3])
    stability = traits["stability"]      # 安定感 (近似 z[2] / z[3] の負の相関)
    energy = traits["energy"]            # 音量・活動性 (近似 z[4])
    tempo_index = traits["tempo_index"]  # テンポ (近似 z[5])
    
    # 揺らぎ、ポーズ、過度な抑揚、テンポ安定性を以下の特性から推定
    # TENSION (緊張感) が高いほど揺らぎ、ポーズ過多、過度な抑揚がある傾向と仮定
    # STABILITY (安定感) が高いほど揺らぎがなく、テンポが安定していると仮定
    
    # 揺らぎ(z[2]の負) = stability
    # 過度な抑揚(z[3]の負) = stability
    # ポーズ(z[6]の負) = tempo_index
    # テンポ安定性(z[7]) = stability 
    
    # 1. 外向性 (Extraversion: 社交性・積極性)
    # 抑揚↑(0.35) + 音量↑(0.30) + 話速↑(0.25) - ポーズ↓(0.25)
    ext = (
        0.35 * pitch_var            # 抑揚 (z[1])
        + 0.30 * energy             # 元気さ/音量 (z[4])
        + 0.25 * tempo_index        # テンポ (z[5])
        + 0.25 * tempo_index        # ポーズ↓をテンポ↑で代用 (z[6])
    )

    # 2. 協調性 (Agreeableness: 優しさ・協調的)
    # 揺らぎ↓(-0.35) - 過度な抑揚↓(-0.25) + 抑揚↑(0.20)
    agr = (
        0.35 * stability            # 揺らぎ↓を安定感↑で代用 (-z[2])
        + 0.25 * stability          # 過度な抑揚↓を安定感↑で代用 (-z[3])
        + 0.20 * pitch_var          # 抑揚 (z[1])
    )
    
    # 3. 誠実性 (Conscientiousness: 計画性・責任感)
    # ポーズ過多↓(-0.30) + テンポ安定↑(0.25) - 揺らぎ↓(-0.20)
    con = (
        0.30 * tempo_index          # ポーズ過多↓をテンポ↑で代用 (-z[6])
        + 0.25 * stability          # テンポ安定↑を安定感↑で代用 (z[7])
        + 0.20 * stability          # 揺らぎ↓を安定感↑で代用 (-z[2])
    )

    # 4. 情緒安定性 (Emotional_Stability: 不安の少なさ・平静さ)
    # 抑揚過多↓(-0.30) - 揺らぎ↓(-0.30) - 過度な抑揚↓(-0.20) + 適度な間↑(0.15)
    sta = (
        0.30 * (1.0 - pitch_var)    # 抑揚過多↓を抑揚↓で代用 (-z[1])
        + 0.30 * stability          # 揺らぎ↓を安定感↑で代用 (-z[2])
        + 0.20 * stability          # 過度な抑揚↓を安定感↑で代用 (-z[3])
        + 0.15 * (1.0 - tempo_index) # 適度な間↑を低テンポで代用 (z[6])
    )
    
    # 5. 開放性 (Openness: 知的好奇心・創造性)
    # 抑揚↑(0.30) + テンポやや速↑(0.25)
    ope = (
        0.30 * pitch_var            # 抑揚 (z[1])
        + 0.25 * tempo_index        # テンポ (z[7])
    )

    # 0.0〜1.0 スケールに戻してクリッピングし、少数第一位に丸める
    # (各スコアを合計し、最大係数 (例: Extは 1.15) で割って正規化するとより正確だが、今回はシンプルにクリップ)
    
    # Ext/Agr/Con/Sta/Ope の係数の最大値は 1.15 / 0.80 / 0.75 / 0.95 / 0.55
    # ここでは、単純にクリップして丸めます。
    
    return {
        "Extraversion": round(float(np.clip(ext, 0.0, 1.0)), 1),
        "Agreeableness": round(float(np.clip(agr, 0.0, 1.0)), 1),
        "Conscientiousness": round(float(np.clip(con, 0.0, 1.0)), 1),
        "Emotional_Stability": round(float(np.clip(sta, 0.0, 1.0)), 1),
        "Openness": round(float(np.clip(ope, 0.0, 1.0)), 1),
    }

# その他の関数 (`extract_voice_features`, `score_voice_traits`, `main`など) は前回の修正版のままで動作します。

# ========= Streamlit UI =========
def main():


    # CSS記述のデザインの適用
    st.markdown(CUTE_CSS, unsafe_allow_html=True)


    st.set_page_config(page_title="あなたの声がささる人", page_icon="🎧", layout="centered")

    st.title("🎧 あなたの声が刺さる人")
    st.caption(
        "声のテンション・安定感・明るさなどから、"
        "“どんな人があなたの声にハマりやすいか”をざっくり診断するエンタメツールです。"
    )

    st.subheader("性別を選択してください")
    gender = st.radio("あなたの性別は？", ["男性", "女性"], index=1, horizontal=True)

    st.markdown("---")
    st.subheader("1. 声の録音")

    st.write(
        "普段どおりの話し方で、30秒〜1分ほど話した音声がおすすめです。\n"
        "最近あったことや、好きなものについて話して録音すると、結果がそれっぽくなります。\n\n"
        
        "**💡 何を話すか迷ったら、以下の例文を読んでみてください。**\n"
        "> 「健康を維持することは、多くの人にとって重要な課題だと思います。適度な運動、バランスの取れた食事、そして十分な睡眠の三つが基本要素だとされています。分かってはいても、現代の忙しい生活の中で、これらを毎日完璧に実行するのは難しいことかもしれません。少しずつでも意識して取り入れることが大切だと考えます。」"
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
                big_five_scores = score_big_five_traits(traits) 
                fan_info = classify_fan_types(traits, big_five_scores, gender)
                trait_text = explain_traits(traits, big_five_scores) 
            except ValueError as ve:
                trait_text = explain_traits(traits)
            # 【変更点: ValueErrorを捕捉】
            except ValueError as ve:
                if "less than 10 seconds" in str(ve):
                    st.error("❌ **エラー:** 録音時間が短すぎます。正確な診断のため、**10秒以上**話して再録音してください。")
                elif "appears to be silent" in str(ve):
                    st.error("❌ **エラー:** 音声が小さすぎるか、**無音**です。マイクの設定を確認し、話しながら再録音してください。")
                else:
                    st.error(f"分析中にエラーが発生しました: {ve}")
                return
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