# reincarnation_vocation_from_voice_v2.py
# 🎙 声からわかる「来世の適職診断」・細かめ分析版
#
# 世界観：
#   - fantasy   : ファンタジー
#   - occult    : オカルト
#   - dystopia  : デストピア
#   - future    : 近未来
#   - ancient   : 古代
#
# 追加分析：
#   - sensitivity   : 共感・繊細さ
#   - sociality     : 社交性・外向き度
#   - introspection : 内省・一人時間の深さ
#   - playfulness   : 遊び心・おちゃらけ度
#
# 出力項目：
#   2. あなたの声の特性（基礎＋追加指標）
#   3. 来世のあなたの特徴（世界観＋人格・ハイブリッド対応）
#   4. 来世の適職
#   5. 来世のあなたはこんな声（テキスト＋TTS）
#   6. 来世のあなたから今世のあなたへのメッセージ
#   7. 来世のあなたの一日の過ごし方
#   8. 来世の恋愛観
#   9. 来世のあなたの恋人
#  10. 来世のために今できること

import tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st


CUTE_CSS = """
<style>
/* Google Fonts：かっちり × 柔らかの中間 */
/* Google Fonts 読み込み */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@700&family=Shippori+Antique+B1:wght@400;500;600;700&display=swap');

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




# TTS 用
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


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
    """
    声の特徴からざっくり指標を作る：
      - energy        : 元気さ
      - tension       : 緊張感・とがり
      - stability     : 安定感
      - brightness    : 声の明るさ
      - roughness     : ザラつき
      - pitch_var     : 抑揚の大きさ
      - tempo_index   : 話すテンポ
      - sensitivity   : 共感・繊細さ
      - sociality     : 社交性・外向き度
      - introspection : 内省・一人時間の深さ
      - playfulness   : 遊び心・おちゃらけ度
    """
    rms_mean = features["rms_mean"]
    rms_std = features["rms_std"]
    zcr_mean = features["zcr_mean"]
    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_std = features["f0_std"]

    energy = 0.6 * normalize(rms_mean, 0.005, 0.05) + 0.4 * normalize(tempo, 60, 180)
    tension = (
        0.4 * normalize(zcr_mean, 0.01, 0.2)
        + 0.3 * normalize(rms_std, 0.0, 0.03)
        + 0.3 * normalize(centroid_mean, 1000, 4000)
    )
    stability = 1.0 - 0.5 * normalize(f0_std, 0, 40) - 0.5 * normalize(rms_std, 0.0, 0.03)
    stability = float(np.clip(stability, 0.0, 1.0))
    brightness = normalize(centroid_mean, 1000, 4000)
    roughness = normalize(zcr_mean, 0.01, 0.2)
    pitch_var = normalize(f0_std, 0, 40)
    tempo_index = normalize(tempo, 60, 180)

    # 追加指標
    sensitivity = (
        0.4 * (1 - tempo_index)   # ゆっくりめほど繊細
        + 0.3 * tension           # 緊張しやすさ＝他人に敏感
        + 0.3 * pitch_var         # 抑揚＝感情の動き
    )

    sociality = (
        0.4 * energy
        + 0.3 * brightness
        + 0.3 * pitch_var
    )

    introspection = (
        0.4 * (1 - tempo_index)
        + 0.3 * (1 - brightness)
        + 0.3 * stability
    )

    playfulness = (
        0.4 * pitch_var
        + 0.3 * brightness
        + 0.3 * (1 - tension)
    )

    return {
        "energy": energy,
        "tension": tension,
        "stability": stability,
        "brightness": brightness,
        "roughness": roughness,
        "pitch_var": pitch_var,
        "tempo_index": tempo_index,
        "sensitivity": float(np.clip(sensitivity, 0.0, 1.0)),
        "sociality": float(np.clip(sociality, 0.0, 1.0)),
        "introspection": float(np.clip(introspection, 0.0, 1.0)),
        "playfulness": float(np.clip(playfulness, 0.0, 1.0)),
    }

# ========= 来世の性別 =========
def infer_next_life_gender(traits: dict, input_gender: str) -> tuple[str, str]:
    """
    声の傾向から「来世の性別傾向」をざっくり推定する。
    ・フェミニン寄り：共感・繊細さ・内省・遊び心・優しさ（低攻撃性）
    ・マスキュリン寄り：行動力・緊張感・タフさ・テンポの速さ
    戻り値：
      next_gender: "女性" / "男性" / "その他・ひみつ"
      reason_text: 説明テキスト
    """
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    ti = traits["tempo_index"]
    sens = traits["sensitivity"]
    intro = traits["introspection"]
    play = traits["playfulness"]
    soc = traits["sociality"]

    # フェミニン寄りスコア（やさしさ・受容・内省・柔らかさ）
    feminine = (
        0.25 * sens +          # 共感・繊細
        0.15 * intro +         # 内省
        0.15 * play +          # 遊び心・クリエイティブ
        0.15 * b +             # 明るさ
        0.15 * (1 - t) +       # 低攻撃性
        0.15 * (1 - r)         # ザラつきの少なさ
    )

    # マスキュリン寄りスコア（行動力・攻撃性・タフさ・スピード）
    masculine = (
        0.25 * e +             # 行動力
        0.2  * t +             # 戦闘モード／緊張感
        0.15 * ti +            # テンポの速さ
        0.15 * r +             # タフさ
        0.15 * (1 - intro) +   # あまり立ち止まらない
        0.1  * soc             # 外向きさ
    )

    feminine = float(np.clip(feminine, 0.0, 1.0))
    masculine = float(np.clip(masculine, 0.0, 1.0))

    diff = feminine - masculine
    # どちらかにそこそこ差があるときだけ「来世側の性別」をスイッチ
    threshold = 0.12

    # デフォルトは入力性別
    if input_gender in ["女性", "男性"]:
        base_gender = input_gender
    else:
        base_gender = "その他・ひみつ"

    if diff > threshold:
        # フェミニン寄りにかなり傾いている
        next_gender = "女性"
    elif diff < -threshold:
        # マスキュリン寄りにかなり傾いている
        next_gender = "男性"
    else:
        next_gender = base_gender

    # 説明テキスト
    if next_gender == base_gender:
        reason = (
            f"・声の傾向からみたフェミニン度：{feminine:.2f} / "
            f"マスキュリン度：{masculine:.2f}\n"
            "・どちらか一方に極端には振れていないため、"
            "来世の性別も今世と近いバランスで引き継がれそうです。"
        )
    else:
        if next_gender == "女性" and base_gender == "男性":
            reason = (
                f"・声の傾向からみたフェミニン度：{feminine:.2f} / "
                f"マスキュリン度：{masculine:.2f}\n"
                "・共感性や柔らかさ、内省の深さがやや優勢なため、"
                "来世では“女性性”が前面に出る生まれ変わりになりそうです。"
            )
        elif next_gender == "男性" and base_gender == "女性":
            reason = (
                f"・声の傾向からみたフェミニン度：{feminine:.2f} / "
                f"マスキュリン度：{masculine:.2f}\n"
                "・行動力やタフさ、スピード感がやや優勢なため、"
                "来世では“男性性”が前面に出る生まれ変わりになりそうです。"
            )
        else:
            reason = (
                f"・声の傾向からみたフェミニン度：{feminine:.2f} / "
                f"マスキュリン度：{masculine:.2f}\n"
                "・声の持つエネルギーの配分から、来世では性別の枠をゆるくまたいだ存在になりそうです。"
            )

    reason += "\n\n※あくまで声の雰囲気からの“エンタメ的な推定”であり、"
    reason += "現実のジェンダーやアイデンティティを決めつけるものではありません。"

    return next_gender, reason


# ========= 世界観スコア：ファンタジー／オカルト／デストピア／近未来／古代 =========
def score_world_axes(traits: dict) -> dict:
    """
    声から世界観の軸をざっくり算出：
      - fantasy  : ファンタジー度
      - occult   : オカルト度
      - dystopia : デストピア度
      - future   : 近未来度
      - ancient  : 古代度
    """
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]
    intro = traits["introspection"]
    play = traits["playfulness"]

    # 明るくて抑揚があり、緊張低め・遊び心あり → ファンタジー寄り
    fantasy = (
        0.3 * b
        + 0.25 * pv
        + 0.2 * play
        + 0.15 * (1 - t)
        + 0.1 * e
    )

    # 暗め・ゆっくり・内省強め・少しザラつき → オカルト寄り
    occult = (
        0.25 * (1 - b)
        + 0.25 * intro
        + 0.2 * (1 - ti)
        + 0.15 * r
        + 0.15 * t
    )

    # ザラつき・緊張・不安定・暗め → デストピア寄り
    dystopia = (
        0.3 * r
        + 0.25 * t
        + 0.2 * (1 - s)
        + 0.15 * (1 - b)
        + 0.1 * e
    )

    # 近未来：明るめ・テンポ速め・安定中くらい・社交性高め
    future = (
        0.25 * b
        + 0.25 * ti
        + 0.2 * traits["sociality"]
        + 0.15 * (1 - abs(s - 0.5) * 2)
        + 0.15 * (1 - r)
    )

    # 古代：テンポ遅め・暗め〜中間・安定高め・内省高め
    ancient = (
        0.25 * (1 - ti)
        + 0.25 * (1 - b)
        + 0.25 * s
        + 0.25 * intro
    )

    fantasy = float(np.clip(fantasy, 0.0, 1.0))
    occult = float(np.clip(occult, 0.0, 1.0))
    dystopia = float(np.clip(dystopia, 0.0, 1.0))
    future = float(np.clip(future, 0.0, 1.0))
    ancient = float(np.clip(ancient, 0.0, 1.0))

    return {
        "fantasy": fantasy,
        "occult": occult,
        "dystopia": dystopia,
        "future": future,
        "ancient": ancient,
    }


# ========= 声の特性の説明 =========
def describe_voice_traits(traits: dict, world_scores: dict) -> str:
    base_labels = {
        "energy": "元気さ",
        "tension": "緊張感・とがり",
        "stability": "安定感",
        "brightness": "声の明るさ",
        "roughness": "ザラつき",
        "pitch_var": "抑揚の大きさ",
        "tempo_index": "話すテンポ",
    }
    extra_labels = {
        "sensitivity": "共感・繊細さ",
        "sociality": "社交性・外向き度",
        "introspection": "内省・一人時間の深さ",
        "playfulness": "遊び心・おちゃらけ度",
    }

    lines = []
    lines.append("● あなたの声の主なパラメータ（0〜1のざっくり指標）")
    for k in base_labels:
        lines.append(f"- {base_labels[k]} : {traits[k]:.2f}")
    lines.append("")
    lines.append("● 追加で読み取れた性質（0〜1）")
    for k in extra_labels:
        lines.append(f"- {extra_labels[k]} : {traits[k]:.2f}")
    lines.append("")

    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]
    sens = traits["sensitivity"]
    soc = traits["sociality"]
    intro = traits["introspection"]
    play = traits["playfulness"]

    # 基本的なコメント
    if e > 0.65:
        lines.append("・声全体にハリがあり、エネルギッシュな印象が強めです。")
    elif e < 0.35:
        lines.append("・少し省エネ気味の声。落ち着きや静けさを含んだトーンです。")

    if t > 0.6:
        lines.append("・言葉の端に少しピリッとした緊張感や鋭さがにじむタイプです。")
    elif t < 0.4:
        lines.append("・余白を感じる、やわらかめの空気感をまとった声です。")

    if s > 0.7:
        lines.append("・安定感が高く、聞き手に安心感を与えやすい声です。")
    elif s < 0.4:
        lines.append("・その時々の気持ちが声に乗りやすい、“揺れ”のあるタイプです。")

    if b > 0.65:
        lines.append("・音色としては明るく、ライトが当たっているような声色です。")
    elif b < 0.4:
        lines.append("・やや暗め・低めに感じられ、深みや影を含んだ声色です。")

    if pv > 0.6:
        lines.append("・抑揚が大きく、感情を表現する力が強い声です。")
    elif pv < 0.4:
        lines.append("・一定のトーンで話すことが多く、落ち着いた印象を与えます。")

    if ti > 0.65:
        lines.append("・話すテンポはやや速めで、キレのある印象です。")
    elif ti < 0.4:
        lines.append("・話すテンポはゆったりめで、時間をゆっくり感じさせる話し方です。")

    # 追加指標ベースのコメント
    if sens > 0.6:
        lines.append("・相手の反応や空気にかなり敏感で、“ちょっとした変化”を拾いやすいタイプです。")
    elif sens < 0.4:
        lines.append("・良い意味で鈍感力があり、多少のことでは動じにくい一面があります。")

    if soc > 0.6:
        lines.append("・人と関わるモードに入ると、場の空気を動かす側に回りやすいです。")
    elif soc < 0.4:
        lines.append("・無理に前に出るより、“少人数”や“一対一”でじっくり話す方が力を発揮しやすいタイプです。")

    if intro > 0.6:
        lines.append("・一人で考えごとをする時間が、心のメンテナンスになりやすい人です。")
    elif intro < 0.4:
        lines.append("・あまり考え込まず、とりあえず動きながら整えていくタイプでもあります。")

    if play > 0.6:
        lines.append("・ちょっとした冗談や遊び心を挟める、“場をゆるめる”声の持ち主です。")
    elif play < 0.4:
        lines.append("・真面目寄りのトーンで、話に重みを持たせやすい声です。")

    lines.append("")
    lines.append("● 世界観スコア（0〜1）")
    for k, v in world_scores.items():
        label = {
            "fantasy": "ファンタジー度",
            "occult": "オカルト度",
            "dystopia": "デストピア度",
            "future": "近未来度",
            "ancient": "古代度",
        }[k]
        lines.append(f"- {label} : {v:.2f}")

    lines.append("")
    lines.append("※これはエンタメ用の簡易指標です。医学的・専門的な診断ではありません。")

    return "\n".join(lines)


# ========= 世界観タイプ（ハイブリッド対応） =========
def get_world_type(world_scores: dict) -> tuple[str, str]:
    """
    dominant_type: 分岐用のメイン世界タイプ
    desc_type: 説明文用（ハイブリッド表現含む）
    """
    sorted_worlds = sorted(world_scores.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_worlds[0], sorted_worlds[1]
    w1, v1 = top1
    w2, v2 = top2

    # 閾値：0.12差以内ならハイブリッドと見る
    if v1 >= 0.45 and (v1 - v2) <= 0.12:
        return w1, f"{w1}+{w2}"
    else:
        return w1, w1

# ========= 世界観の説明 =========
def world_desc_from_type(desc_type: str) -> str:
    if "+" in desc_type:
        a, b = desc_type.split("+")
        pair = {a, b}
        if pair == {"fantasy", "future"}:
            return "物語とテクノロジーが同居した世界。少し不思議だけど、生活はわりと実用的です。"
        if pair == {"fantasy", "ancient"}:
            return "古い神話や祭りが今も身近にある世界。日常と物語がゆるくつながっています。"
        if pair == {"future", "dystopia"}:
            return "便利な技術と、少し荒れた環境が同時に存在する近未来の世界です。"
        if pair == {"occult", "ancient"}:
            return "見えないものの存在が普通に受け入れられている、少し昔の世界です。"
        if pair == {"fantasy", "occult"}:
            return "夢や物語と、直感や霊感が近い場所にある世界です。"
        return "二つの世界観がほどよく混ざり合った世界。どちらの雰囲気も少しずつ感じられます。"

    if desc_type == "fantasy":
        return "魔法や物語の要素が、今より少しだけ身近にあるファンタジー寄りの世界です。"
    if desc_type == "occult":
        return "直感や勘、見えないものの気配が少しだけ重視される世界です。"
    if desc_type == "dystopia":
        return "環境はやや厳しめですが、工夫しながら暮らしている人が多い世界です。"
    if desc_type == "future":
        return "テクノロジーが今より進んでいて、生活のあちこちに馴染んでいる近未来の世界です。"
    if desc_type == "ancient":
        return "季節や星の動きが今より身近で、自然と一緒に暮らしている古めの世界です。"
    return "現世とかなり近い、リアル寄りの世界観です。ほんの少しだけ物語っぽさが足されています。"

# ========= 来世のあなたの特徴を作る =========
def build_next_life_profile(traits: dict, world_scores: dict, gender: str) -> str:

    lines: list[str] = []

    # 強い・弱い項目
    max_key = max(traits.items(), key=lambda x: x[1])[0]
    min_key = min(
        [(k, v) for k, v in traits.items()
         if k in ["energy", "tension", "stability", "brightness", "roughness", "pitch_var", "tempo_index"]],
        key=lambda x: x[1]
    )[0]

    label_map = {
        "energy": "行動力・生命力",
        "tension": "緊張感・警戒心",
        "stability": "心の安定力",
        "brightness": "雰囲気の明るさ",
        "roughness": "タフさ・サバイバル力",
        "pitch_var": "表現力・感情の振れ幅",
        "tempo_index": "決断スピード・テンポ感",
        "sensitivity": "共感力・繊細さ",
        "sociality": "社交性・外向き度",
        "introspection": "内省の深さ",
        "playfulness": "遊び心・おちゃらけ度",
    }

    max_label = label_map[max_key]
    min_label = label_map[min_key]

    dominant_world, desc_type = get_world_type(world_scores)

    # world_type（分岐用）
    if max(world_scores.values()) < 0.4:
        world_type = "real"
    else:
        world_type = dominant_world

    world_desc = world_desc_from_type(desc_type)

    # ------------------------------
    # 来世の性別（独立項目）
    # ------------------------------
    if gender == "女性":
        gender_phrase = "来世のあなたは“女性”として生まれる可能性が比較的高いようです。"
    elif gender == "男性":
        gender_phrase = "来世のあなたは“男性”として生まれる可能性が比較的高いようです。"
    else:
        gender_phrase = "来世のあなたは、性別の枠をゆるやかにまたぐ存在として生まれる可能性があります。"

    lines = []

    lines.append("● 来世のあなたの性別")
    lines.append(gender_phrase)
    lines.append("")  # 空行（段落区切り）

    
    lines.append("● 来世のあなたが生きる世界観")
    lines.append(world_desc)
    lines.append("")
    lines.append("● 来世のあなたのベース人格")
    lines.append(
        f"- 今の声で特に強く出ているのは「{max_label}」。"
        " 来世ではここが少し伸びて、その世界での“わかりやすい得意分野”になります。"
    )
    lines.append(
        f"- 逆に、今の声で一番控えめだった「{min_label}」は、来世ではゆるく底上げされ、"
        "苦手というより“バランスをとるための補助的な力”になります。"
    )
    lines.append("")

    # 世界観ごとのざっくり人物像
    lines.append("● ざっくりした来世の人物像")

    if world_type == "fantasy":
        lines.append(
            "物語の登場人物のように、周りの人の感情や空気を動かす存在になります。\n"
            f"「{max_label}」を武器に、人の心に火をつけたり、物語を進める役割を担い、"
            f"「{min_label}」が程よく補強されることで、暴走せずに世界と折り合いをつけていけるタイプです。"
        )

    elif world_type == "occult":
        lines.append(
            "人の目には見えない“裏の情報”や、場の気配をすくい取る役割をもつ人物になります。\n"
            f"「{max_label}」が鋭いアンテナとなり、「{min_label}」が底支えすることで、"
            "クセはあるけれど頼られる存在になりそうです。"
        )

    elif world_type == "dystopia":
        lines.append(
            "少し荒れた時代のなかで、“したたかに、でもまっすぐに生きる人”になります。\n"
            f"「{max_label}」が生存戦略の中心となり、「{min_label}」が底上げされることで、"
            "諦めずに未来をつなぐポジションに立ちやすいタイプです。"
        )

    elif world_type == "future":
        lines.append(
            "情報と感情のあいだを行き来しながら、人とテクノロジーの橋渡しをする人物になります。\n"
            f"「{max_label}」を使って、人の心を動かすインターフェースとなり、"
            f"「{min_label}」が整うことで、過剰に振り回されずに時代を乗りこなせるタイプです。"
        )

    elif world_type == "ancient":
        lines.append(
            "祭りや儀式、共同体の真ん中で“声”を使う役割を持つ人物になりやすいです。\n"
            f"「{max_label}」が人々をまとめる力になり、「{min_label}」が底上げされることで、"
            "神話と日常の橋渡しをするようなポジションに立ちます。"
        )

    else:
        lines.append(
            "現世とそこまで変わらない、落ち着いた世界で生きることになりそうです。\n"
            f"「{max_label}」はそのまま強みとして評価され、"
            f"「{min_label}」も今より扱いやすいレベルに整った“ちょうど良いバランス型”の人格になります。"
        )

    
    return "\n".join(lines), world_type

# ========= 来世の適職 =========
def choose_next_life_jobs(traits: dict, world_scores: dict, world_type: str, gender: str) -> dict:
    f = world_scores
    dom_score = max(f.values())

    jobs = []
    explanation = ""

    # 性別のざっくりカテゴリ
    is_female = (gender == "女性")
    is_male = (gender == "男性")

    # どの世界観にも強く振れていないとき
    if dom_score < 0.4 or world_type == "real":
        if is_male:
            jobs = [
                "事業や組織の“裏側配線”を組み立てるストラテジープランナー",
                "人とチームのメンタルを整えるメンタルマネージャー",
                "複数プロジェクトを動かすハイパー・プロジェクトマネージャー",
            ]
        elif is_female:
            jobs = [
                "情報と物語を整理するコンテンツエディター",
                "心を整えるライフデザインカウンセラー",
                "人とプロジェクトをつなぐオーガナイズプロデューサー",
            ]
        else:
            jobs = [
                "構造を整理するストラクチャーデザイナー",
                "心と場を整えるファシリテーター",
                "静かに企画を動かすプロジェクトデザイナー",
            ]

        explanation = (
            "世界観スコアを見ると、どの世界観も極端に突出してはいません。\n"
            "来世のあなたは、現世にかなり近い、落ち着いた世界で生きる可能性が高めです。\n\n"
            "その中で、声の安定感や抑揚、テンポ感を総合すると、"
            "情報や人間関係の“裏側の配線”を組み立てる仕事との相性が良さそうです。"
        )

    # ファンタジー世界
    elif world_type == "fantasy":
        if is_male:
            jobs = [
                "異世界ギルドのストラテジープランナー",
                "物語バトルのシナリオディレクター",
                "魔法技術と現場をつなぐフィールドコーディネーター",
            ]
        elif is_female:
            jobs = [
                "物語世界のアーカイブキュレーター",
                "夢の設計士（ドリームデザイナー）",
                "精霊と人間のストーリーブリッジャー",
            ]
        else:
            jobs = [
                "物語と現実をつなぐストーリーデザイナー",
                "人の“物語性”を引き出すナラティブコーチ",
                "精霊と人間のコミュニケーションデザイナー",
            ]

        explanation = (
            "ファンタジー度が高く、声に“物語の余白”やイメージの広がりがにじんでいます。\n"
            "来世のあなたは、物語と現実の境界線が今よりゆるい世界で、"
            "物語の流れを組み立てたり、人の夢やイメージを設計したり、"
            "異なる存在同士の“ストーリーの橋渡し”をする役割に強い適性があります。"
        )

    # オカルト世界
    elif world_type == "occult":
        if is_male:
            jobs = [
                "都市伝説インテリジェンスオフィサー",
                "夢と記憶を解析するディープリーディング・アナリスト",
                "未練をほどく“聞き取り特化”スピリット・リスナー",
            ]
        elif is_female:
            jobs = [
                "都市伝説アーカイバー",
                "夢と記憶を扱う占術師",
                "心残りをほどく“聞き取り専門”霊媒",
            ]
        else:
            jobs = [
                "見えない情報を扱うインビジブル・アーカイバー",
                "夢と記憶のストーリーガイド",
                "場の気配を調整するエネルギー・モデレーター",
            ]

        explanation = (
            "オカルト度が高く、声に“見えないものを拾うアンテナ”のような性質が出ています。\n"
            "来世のあなたは、ただ怖がらせるのではなく、"
            "人の記憶・未練・都市伝説の裏側を静かに拾い上げて整理する専門家として生きる素質があります。"
        )

    # デストピア世界
    elif world_type == "dystopia":
        if is_male:
            jobs = [
                "崩れかけた都市のインフォメーションハンター",
                "デジタル廃墟のアーカイブ管理リーダー",
                "サバイバル戦略プランナー",
            ]
        elif is_female:
            jobs = [
                "崩れかけた都市の情報ハンター",
                "デジタル廃墟のアーカイブ管理人",
                "生き延びるためのシナリオ・プランナー",
            ]
        else:
            jobs = [
                "ポストアポカリプス・ストラテジスト",
                "情報と物資のルートデザイナー",
                "サバイバル・コミュニティのナビゲーター",
            ]

        explanation = (
            "デストピア度が高く、声に“タフさ”や“警戒心”が少し滲んでいます。\n"
            "来世のあなたは、完全に希望がない世界ではなく、"
            "ギリギリ希望をつなぐために“情報”や“戦略”を扱う職業に強い適性を持つタイプです。"
        )

    # 近未来世界
    elif world_type == "future":
        if is_male:
            jobs = [
                "ヒトとAIのインターフェースアーキテクト",
                "都市スケールUXプランナー",
                "時間差コミュニケーション・ストラテジスト",
            ]
        elif is_female:
            jobs = [
                "ヒトとAIのインターフェースデザイナー",
                "暮らしの体験を編むライフUXプランナー",
                "オンラインとオフラインをつなぐコミュニケーション調整士",
            ]
        else:
            jobs = [
                "感情とテクノロジーをつなぐエモーション・エンジニア",
                "都市の体験設計ディレクター",
                "時間と情報のフローコーディネーター",
            ]

        explanation = (
            "近未来度が高く、声に“情報と感情の橋渡し”の素質がにじんでいます。\n"
            "来世のあなたは、高度なテクノロジーと人間の心のあいだをつなぎ、"
            "暮らしや都市の体験をデザインする仕事に強い適性がありそうです。"
        )

    # 古代世界
    elif world_type == "ancient":
        if is_male:
            jobs = [
                "祭礼のストーリーテラー",
                "星と暦を読むナビゲーター",
                "村と村をつなぐトラベル・ネゴシエーター",
            ]
        elif is_female:
            jobs = [
                "祭礼の語り部",
                "星と暦を読む案内人",
                "村と村をつなぐ旅の交渉人",
            ]
        else:
            jobs = [
                "共同体の物語ファシリテーター",
                "季節と星を読むリズムガイド",
                "境界を行き来するトラベル・メディエーター",
            ]

        explanation = (
            "古代度が高く、声に“時間の流れをゆっくり感じさせる性質”が出ています。\n"
            "来世のあなたは、人々を集めて物語を語ったり、星や季節を読み取って、"
            "共同体の進む方向をそっと示すような仕事に縁がありそうです。"
        )

    else:
        # 念のためのフォールバック（ほぼ来ない想定）
        jobs = ["構造を整理する編集者・ライター", "心を整えるカウンセラー", "静かな企画屋・プロデューサー"]
        explanation = (
            "世界観は現実寄りで、極端な物語世界ではない分、"
            "“現実の中で物語を見つける仕事”との相性が良さそうです。"
        )

    main_job = jobs[0]
    sub_jobs = jobs[1:]

    return {
        "main_job": main_job,
        "sub_jobs": sub_jobs,
        "explanation": explanation,
    }


# ========= 来世のために今できること =========
def advice_for_now(traits: dict, world_scores: dict) -> str:
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]
    sens = traits["sensitivity"]
    soc = traits["sociality"]
    intro = traits["introspection"]
    play = traits["playfulness"]

    f = world_scores
    fantasy = f["fantasy"]
    occult = f["occult"]
    dystopia = f["dystopia"]
    future = f["future"]
    ancient = f["ancient"]

    lines = []
    lines.append("● 来世のために、現世のあなたができる具体的なこと")
    lines.append("※全部やる必要はありません。ピンときたものだけ拾ってください。")
    lines.append("")

    if e < 0.4:
        lines.append("・疲れやすさや省エネモードを感じたら、“何もしていない時間”を罪悪感なしで確保する練習をしてみましょう。")
    elif e > 0.7:
        lines.append("・元気が有り余っている日は、ただ動くだけでなく“続けたいこと”に少し多めにエネルギーを振る癖をつけておくと、来世での武器になります。")

    if s < 0.4:
        lines.append("・心が揺れやすいと感じるときは、日記やメモで“気持ちのログ”を残す習慣を。来世での安定力の土台になります。")
    elif s > 0.7:
        lines.append("・安定感が高いあなたは、周りの人の相談に“1つだけ視点を足す”練習をしておくと、来世での支援スキルがより伸びます。")

    if pv > 0.6:
        lines.append("・感情表現が豊かな声なので、言葉以外の表現（文章・絵・写真・音楽など）も1つ持っておくと、来世の表現職に直結しやすくなります。")
    elif pv < 0.4:
        lines.append("・抑揚が少ないと感じるときは、“自分の好き嫌い”を小さくでも口に出す練習をすると、来世での自己主張力の種になります。")

    if sens > 0.6:
        lines.append("・繊細さが強いぶん、情報の取りすぎ注意。意識的に“見ない日”“オフの時間”を作っておくと、来世の感度がちょうど良くなります。")
    elif sens < 0.4:
        lines.append("・鈍感力があるので、時々だけ“自分以外の人の1日”を想像してみる練習をすると、来世での共感力の幅が広がります。")

    if soc > 0.6:
        lines.append("・社交性が高めなので、“ただ一緒にいるだけの時間”を意図的に作ると、来世での人間関係がさらにラクになります。")
    elif soc < 0.4:
        lines.append("・社交性が控えめな分、“この人だけは話しやすい”という人を1人キープしておくと、来世での安心できる拠点になります。")

    if intro > 0.6:
        lines.append("・内省が深いあなたは、メモやノートに“気づき”だけ書き溜めておくと、来世での洞察力の基礎データになります。")
    elif intro < 0.4:
        lines.append("・そこまで考え込まない性質なので、たまにだけ“今日一番うれしかったこと”を3行以内で書くと、来世のポジティブさが底上げされます。")

    if play > 0.6:
        lines.append("・遊び心が強いので、“ふざけていいライン”を自分なりに決めておくと、来世でも人間関係を壊さずに場を和ませられます。")
    elif play < 0.4:
        lines.append("・真面目寄りなぶん、“自分だけの小さな笑えるネタ帳”を作ると、来世のユーモア筋が鍛えられます。")

    if fantasy > max(occult, dystopia, future, ancient) and fantasy > 0.5:
        lines.append("・ファンタジー度が高めなので、「好きな物語の世界」をノートに書き留めておくと、来世の世界設定の材料になりやすいです。")
    if occult > max(fantasy, dystopia, future, ancient) and occult > 0.5:
        lines.append("・オカルト度が高めなので、直感メモ（日付と“なんとなくこう思った”を一行だけ書く）を続けると、来世の“感覚の精度”が上がります。")
    if dystopia > max(fantasy, occult, future, ancient) and dystopia > 0.5:
        lines.append("・デストピア度が高めなあなたは、“困ったときに頼れるリスト”（人・場所・サービス）を作っておくと、来世のサバイバル知恵の原型になります。")
    if future > max(fantasy, occult, dystopia, ancient) and future > 0.5:
        lines.append("・近未来度が高めなので、新しいツールやサービスを試したときの感想をメモしておくと、来世の“使いこなしスキル”の土台になります。")
    if ancient > max(fantasy, occult, dystopia, future) and ancient > 0.5:
        lines.append("・古代度が高めなので、神話・歴史・民話の中で「好きな人物」や「好きな物語」を1つ決めておくと、来世での役割選びのヒントになります。")

    if all(x <= 0.5 for x in [fantasy, occult, dystopia, future, ancient]):
        lines.append("・世界観が現実寄りなので、“今日やってよかったことを3つだけ書く”習慣を持つと、来世の基礎ステータスがじわじわ上がります。")

    return "\n".join(lines)


# ========= 来世の声のイメージ／ボイスサンプル・メッセージ =========
def describe_next_life_voice(traits: dict, world_type: str) -> str:
    b = traits["brightness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]
    s = traits["stability"]

    lines = []
    lines.append("● 来世のあなたの“声質イメージ”")

    if world_type == "fantasy":
        lines.append("・少し高めで透明感があり、言葉の端に“きらっとした光”を感じる声。")
    elif world_type == "occult":
        lines.append("・やや低めで、ささやくような響きがある、静かな深みのある声。")
    elif world_type == "dystopia":
        lines.append("・少しザラっとした質感を持ちながらも、芯の強さが伝わる声。")
    elif world_type == "future":
        lines.append("・クリアでノイズが少なく、電子音と生声の中間のような、滑らかな声。")
    elif world_type == "ancient":
        lines.append("・土や木の匂いを連想させるような、あたたかく厚みのある声。")
    else:
        lines.append("・現実世界にもいそうな、落ち着いたトーンの話しやすい声。")

    if b > 0.6 and pv > 0.6:
        lines.append("・感情の起伏が声にしっかり乗り、聞く人の心にイメージを直接流し込むタイプです。")
    elif b > 0.6 and pv <= 0.6:
        lines.append("・明るい音色ながら、起伏は穏やかで、安らぎを与える性質があります。")
    elif b <= 0.6 and pv > 0.6:
        lines.append("・落ち着いた色味の声に、ところどころ強いアクセントが入る、印象に残りやすい声です。")
    else:
        lines.append("・トーンも抑揚も穏やかで、BGMのようにそっと寄り添う声です。")

    if ti > 0.65:
        lines.append("・話すテンポはやや速めで、セリフが“矢のようにスッと飛んでいく”印象があります。")
    elif ti < 0.4:
        lines.append("・テンポはゆっくりめで、時間が少しだけスローモーションに感じられる話し方です。")

    if s > 0.7:
        lines.append("・安定感が高く、長く聞いていても疲れにくい、信頼されやすい声です。")
    elif s < 0.4:
        lines.append("・感情の揺れや迷いが、いい意味でそのまま声に乗りやすい、ライブ感のある声です。")

    return "\n".join(lines)



# ========= 来世の自分からのメッセージ（声） =========
def build_next_life_voice_sample_text(world_type: str, gender: str) -> str:
    if gender == "女性":
        me_word = "来世のわたし"
    elif gender == "男性":
        me_word = "来世のぼく"
    else:
        me_word = "来世のわたし"

    if world_type == "fantasy":
        text = (
            f"こんにちは、{me_word}です。物語と現実のあいだを行き来しながら暮らしている、少し先のあなたです。"
            " 今世のあなたが選んだ小さな一歩も、ちゃんとこちらに届いています。"
        )
    elif world_type == "occult":
        text = (
            f"聞こえていますか、{me_word}からのメッセージです。"
            " あなたがふと感じる嫌な予感や、なぜか安心できる場所には、ちゃんと理由があります。"
        )
    elif world_type == "dystopia":
        text = (
            f"やあ、{me_word}だよ。少し荒れた世界で、それでも普通に生活している、もう一人のあなたです。"
            " 今のあなたの工夫や我慢は、こちらではサバイバルの知恵として役立っています。"
        )
    elif world_type == "future":
        text = (
            f"こんにちは、{me_word}です。透明な画面越しに、今世のあなたを見ています。"
            " あなたが新しいものに少しずつ慣れていこうとしている姿勢が、こちらでは“扱い方の基礎”になっています。"
        )
    elif world_type == "ancient":
        text = (
            f"もしもし、{me_word}です。焚き火のそばから、あなたに話しかけています。"
            " あなたが今日、誰かの話を最後まで聞いたことは、こちらの世界の物語をそっと支えています。"
        )
    else:
        text = (
            f"こんにちは、{me_word}です。現世とよく似た世界で暮らしている、少し先のあなたです。"
            " 今日もちゃんと起きて、ご飯を食べて、人と関わっているだけで、こちらの土台はだいぶ整っています。"
        )

    return text


# ========= 来世の自分からのメッセージ（テキスト） =========

def build_message_from_next_life(traits: dict, world_type: str, gender: str) -> str:
    e = traits["energy"]
    s = traits["stability"]
    t = traits["tension"]

    if gender == "女性":
        you_call = "今のわたしへ"
    elif gender == "男性":
        you_call = "今のぼくへ"
    else:
        you_call = "今のあなたへ"

    lines = []
    lines.append(f"『{you_call}』")
    lines.append("")

    if world_type == "fantasy":
        lines.append("あなたがときどき“現実的じゃないかな…”と思って胸の奥にしまったアイデアや妄想は、こっちの世界ではちゃんと仕事になっているよ。")
        lines.append("あのときノートに書きかけてやめた設定、あのまま忘れていないから安心して。")
    elif world_type == "occult":
        lines.append("あなたが“なんとなくそう思った”という直感は、こっちの世界では立派な情報源になっている。")
        lines.append("空気を読みすぎて疲れた日のことも、全部こっちで“感度の高いレーダー”として再利用しているからムダじゃないよ。")
    elif world_type == "dystopia":
        lines.append("あなたが今、理不尽さや窮屈さに耐えながら身につけている工夫は、こっちの世界では生存戦略としてかなり役に立っている。")
        lines.append("だから、“あのとき我慢した自分”を、もう少しだけ評価してあげてほしい。")
    elif world_type == "future":
        lines.append("あなたが“ついていけないかも”と思いながらも新しいものを試している姿勢が、こっちの世界では“アップデートを恐れない人”という評価になっているよ。")
        lines.append("少しだけでも興味を持ったものに手を伸ばしてくれて、ここのわたしはとても助かっています。")
    elif world_type == "ancient":
        lines.append("あなたが誰かの話をじっくり聞いた時間や、意味もなく空を眺めていた時間は、こっちの世界では“祈りの時間”として蓄積されている。")
        lines.append("だから、役に立たないように見える静かな時間も、そのまま大事にしていてほしい。")
    else:
        lines.append("あなたが“こんな平凡でいいのかな”と思う日々も、こっちから見るとかなり尊い毎日です。")
        lines.append("今日ちゃんと寝て、ご飯を食べて、人と会話していること自体が、次の世界のベースの体力になっているからね。")

    lines.append("")

    if e < 0.4:
        lines.append("もし最近、前よりも声が小さくなった気がしていたら、それはサボりじゃなくて“チャージ中”だから大丈夫。")
    elif e > 0.7:
        lines.append("逆に、ついつい走り続けてしまうときは、「今日はここまで」と決める練習をしておいてくれると、来世のわたしがかなり助かります。")

    if s < 0.4:
        lines.append("心が揺れやすい自分を、ダメだと思わなくていい。揺れたぶんだけ、他の人の揺れにも気づけるから。")
    elif s > 0.7:
        lines.append("安定しているように見えるあなたも、ときどき誰かに弱音を見せておいてくれると嬉しい。こっちの世界では、その“人に頼る練習”がかなり役立っています。")

    if t > 0.6:
        lines.append("いつも少し身構えてしまう自分も、そのおかげで守れているものがたくさんあるよ。たまにだけ、肩と jaw をゆるめて深呼吸してみてほしい。")
    else:
        lines.append("構えすぎないで生きているあなたの感覚は、来世での“しなやかさ”としてちゃんと引き継いでおくね。")

    lines.append("")
    lines.append("最後にひとこと。")
    lines.append("“今日はイマイチだったな”と思う日も、こちらから見るとちゃんと意味のある一日です。少し気楽に過ごしても大丈夫です。")

    return "\n".join(lines)

def generate_next_life_voice_audio(text: str) -> str | None:
    if not GTTS_AVAILABLE:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(text=text, lang="ja")
            tts.save(tmp.name)
            return tmp.name
    except Exception:
        return None




# ========= 来世の恋愛観 =========
def describe_next_life_love_style(traits: dict, world_type: str, next_gender: str) -> str:
    """
    来世の恋愛観を性別と世界観の組み合わせで分岐。
    男性・女性でニュアンスを変える。
    """

    e = traits["energy"]
    s = traits["stability"]
    b = traits["brightness"]
    sens = traits["sensitivity"]
    intro = traits["introspection"]
    play = traits["playfulness"]

    lines = []
    lines.append("● 来世のあなたの恋愛観")

    # ----------------------------------
    # 性別によるベーススタイル
    # ----------------------------------
    if next_gender == "男性":
        base = "来世のあなたは“包容×行動力”型の恋愛スタイルです。"
        if e > 0.6:
            base += "\n・主導権を握る瞬間がありつつ、相手のペースも尊重できるタイプ。"
        if intro > 0.6:
            base += "\n・口数は多くなくても、行動で気持ちを示す派。"
        if sens > 0.6:
            base += "\n・相手の表情の変化をよく見ていて、フォローが自然にできる。"

    elif next_gender == "女性":
        base = "来世のあなたは“直感×心の深さ”型の恋愛スタイルです。"
        if sens > 0.6:
            base += "\n・相手の言葉より“温度”を読むのが得意。"
        if play > 0.6:
            base += "\n・甘えと茶目っ気が混ざった独特の魅力がある。"
        if intro > 0.6:
            base += "\n・言葉は控えめでも、気持ちはじんわり深い。"

    else:
        base = "来世のあなたは“心の距離を柔らかく調整する恋愛スタイル”です。"
        if sens > 0.6:
            base += "\n・とにかく相手の世界を丁寧に扱える人。"
        if play > 0.6:
            base += "\n・相手をほぐすユーモアが自然に出るタイプ。"

    lines.append(base)

    # ----------------------------------
    # 世界観による追加要素
    # ----------------------------------
    if world_type == "fantasy":
        lines.append("・恋愛には“物語性”を求めるタイプ。出会いの瞬間や思い出を大切にする。")
    elif world_type == "occult":
        lines.append("・普通の人が気づかない“感情の裏側”を敏感に察し、寄り添う恋愛をする。")
    elif world_type == "dystopia":
        lines.append("・不安定な世界でも『一緒に生き延びる相手』を選ぶ、強い絆重視タイプ。")
    elif world_type == "future":
        lines.append("・物理距離よりも“情報と感情のやり取り”を大事にする、先進的な恋愛観。")
    elif world_type == "ancient":
        lines.append("・四季や星の巡りとともに“自然な流れで深まる恋”を選びがち。")

    return "\n".join(lines)

# ========= 来世のあなたの恋人像 =========
def describe_next_life_partner(traits: dict, world_type: str, next_gender: str) -> str:
    """
    来世の恋人を“演出控えめでさらっと読める”診断にしたバージョン。
    ・外見や家族構成は簡潔に
    ・性格は traits に基づくが、短めに
    ・あなたの声のどこに惹かれたかも簡潔に
    """

    e = traits["energy"]
    s = traits["stability"]
    b = traits["brightness"]
    intro = traits["introspection"]
    sens = traits["sensitivity"]
    play = traits["playfulness"]
    soc = traits["sociality"]
    r = traits["roughness"]
    ti = traits["tempo_index"]
    pv = traits["pitch_var"]

    lines: list[str] = []
    lines.append("● 来世のあなたの恋人はこんな人")

    # --------------------------------------------
    # 世界観別の雰囲気（さらっと）
    # --------------------------------------------
    if world_type == "fantasy":
        world_note = "自然や物語とのつながりが強い環境で育った人です。"
    elif world_type == "occult":
        world_note = "直感や空気を読む力が育つ環境で暮らしていた人です。"
    elif world_type == "dystopia":
        world_note = "ややシビアな環境をくぐり抜けてきた、少したくましいタイプです。"
    elif world_type == "future":
        world_note = "テクノロジーと共存する都市で育ち、情報感度が高いタイプです。"
    elif world_type == "ancient":
        world_note = "自然のリズムを大切にする共同体で育った、穏やかな人です。"
    else:
        world_note = "落ち着いた家庭環境で育った、現実寄りのタイプです。"

    lines.append(f"・出身の雰囲気：{world_note}")

    # --------------------------------------------
    # 外見（シンプルに）※来世のあなたの性別に応じて反転
    # --------------------------------------------
    if next_gender == "男性":
        # あなたが男性 → 恋人は女性
        appearance = "柔らかい雰囲気の女性。姿勢がきれいで、動きがしなやかです。"
    else:
        # あなたが女性 or その他 → 恋人は男性
        appearance = "穏やかで落ち着いた雰囲気の男性。体型は標準〜やや引き締まりです。"

    lines.append(f"・外見：{appearance}")

    # --------------------------------------------
    # 家庭環境（世界観ベースで簡単に）
    # --------------------------------------------
    if world_type in ["fantasy", "ancient"]:
        family = "兄弟姉妹の多い家庭で育ち、人との関わりが自然な人。"
    elif world_type == "future":
        family = "小さな家族単位で育ち、自立心が強いタイプ。"
    elif world_type == "dystopia":
        family = "家族関係は少し複雑で、自分のペースで生きてきた人。"
    else:
        family = "一般的な家庭環境で育った、バランスの良いタイプ。"

    lines.append(f"・家庭環境：{family}")

    # --------------------------------------------
    # 性格（traits を使って“3つだけ”簡潔に）
    # --------------------------------------------
    personality_list: list[str] = []

    if next_gender == "男性":
        # あなたが男性 → 恋人は女性
        if sens > 0.6:
            personality_list.append("相手の気持ちを汲むのが上手で、柔らかい雰囲気のある人")
        if intro > 0.6:
            personality_list.append("静かで丁寧に関わってくれるタイプ")
        if play > 0.6:
            personality_list.append("さりげなく冗談を交えてくれる、可愛らしい一面がある人")
        if e > 0.6:
            personality_list.append("明るさと行動力のバランスがちょうど良い人")
        if s > 0.7:
            personality_list.append("感情の波が穏やかで、一緒にいると安心できる人")
    else:
        # あなたが女性 or その他 → 恋人は男性
        if e > 0.6:
            personality_list.append("行動力があり、頼りになる面があるタイプ")
        if s > 0.7:
            personality_list.append("落ち着きがあって、安心感を与えてくれる人")
        if intro > 0.6:
            personality_list.append("静かで一緒に穏やかな時間を過ごせる人")
        if sens > 0.6:
            personality_list.append("気持ちの変化によく気づく、思いやりのある人")
        if play > 0.6:
            personality_list.append("ユーモアが適度にあり、話しやすい雰囲気の人")

    if not personality_list:
        if next_gender == "男性":
            personality_list.append("ほどよく落ち着いた、親しみやすい性格の人")
        else:
            personality_list.append("バランスの良い、穏やかで誠実な性格の人")

    lines.append("・性格：")
    for p in personality_list[:3]:
        lines.append(f"  - {p}")

    # --------------------------------------------
    # あなたの声のどこに惹かれたか（簡潔に）
    # --------------------------------------------
    attraction_list: list[str] = []

    if next_gender == "男性":
        # あなたが男性 → 恋人は女性 → 女性が惹かれやすいポイント
        if b > 0.6:
            attraction_list.append("声の明るさと、話しかけやすい雰囲気")
        else:
            attraction_list.append("落ち着いた声の響きと、安心できるトーン")

        if sens > 0.6:
            attraction_list.append("声にのる感情の繊細なゆらぎ")
        if r < 0.4:
            attraction_list.append("耳に残るやわらかい声質")
        if s > 0.7:
            attraction_list.append("安定した声が生み出す安心感")
        if intro > 0.6:
            attraction_list.append("言葉の奥にある“静かな優しさ”")
    else:
        # あなたが女性 or その他 → 恋人は男性 → 男性が惹かれやすいポイント
        if b > 0.6:
            attraction_list.append("明るくて、気分を切り替えてくれる声")
        else:
            attraction_list.append("落ち着いた響きで、話しやすいペースの声")

        if e > 0.6:
            attraction_list.append("前向きさを感じさせる声のハリ")
        if sens > 0.6:
            attraction_list.append("気持ちの動きが素直に伝わるところ")
        if s > 0.7:
            attraction_list.append("ぶれにくく、安心できる声の安定感")
        if pv > 0.6:
            attraction_list.append("抑揚のある話し方で、印象に残るところ")

    if not attraction_list:
        attraction_list.append("なんとなく耳に残る、不思議と気になる声の質感")

    lines.append("・相手があなたに惹かれたポイント：")
    for a in attraction_list[:3]:
        lines.append(f"  - {a}")

    return "\n".join(lines)

# ========= 来世のあなたの似顔絵（イメージ） =========
def describe_next_life_self_portrait(traits: dict, world_type: str, next_gender: str) -> str:
    e = traits["energy"]
    s = traits["stability"]
    b = traits["brightness"]
    intro = traits["introspection"]
    play = traits["playfulness"]

    lines = []
    lines.append("● 来世のあなたの似顔絵（イメージメモ）")

    # 性別ざっくり
    if next_gender == "男性":
        base_face = "やわらかい目元の男性"
    elif next_gender == "女性":
        base_face = "落ち着いた雰囲気の女性"
    else:
        base_face = "中性的な雰囲気の人"

    # 世界観ごとの背景・服装
    if world_type == "fantasy":
        bg = "薄い光が差し込む森や、小さな光の粒が舞う背景"
        outfit = "ゆるやかなローブやケープ風の服装"
    elif world_type == "occult":
        bg = "少し暗めの室内や、月明かりの窓辺"
        outfit = "シンプルなシャツやコートに、さりげないアクセサリー"
    elif world_type == "dystopia":
        bg = "少し荒れた街並みや、古びた看板のある路地"
        outfit = "動きやすいジャケットや、少し使い込まれた服装"
    elif world_type == "future":
        bg = "シンプルな光のラインが入った近未来的な空間"
        outfit = "すっきりしたラインのジャケットや、ミニマルな服装"
    elif world_type == "ancient":
        bg = "石畳や木々、焚き火など、自然や素朴な建物がある場所"
        outfit = "ゆったりした布の服や、民族衣装風の装い"
    else:
        bg = "日常の街角やカフェの一角"
        outfit = "シンプルで普段着寄りのカジュアルな服装"

    # 表情・ポーズ
    if e > 0.6:
        pose = "少し前のめり気味で、目線に動きがある表情"
    elif intro > 0.6:
        pose = "本やノートに視線を落としつつ、少しだけこちらを気にしている表情"
    else:
        pose = "自然体で正面を向き、ほどよく力の抜けた表情"

    if play > 0.6:
        face_detail = "口元にうっすら笑みがあって、いたずらっぽさが少しにじむ"
    elif s > 0.7:
        face_detail = "眉や口元が安定していて、落ち着いた印象が強い"
    else:
        face_detail = "そのときの気分が少し表情に出ている、動きのある顔つき"

    lines.append(f"- 顔立ちのイメージ：{base_face}")
    lines.append(f"- 背景：{bg}")
    lines.append(f"- 服装：{outfit}")
    lines.append(f"- 表情・ポーズ：{pose}、{face_detail}")

    # 小物
    item = None
    if world_type in ["fantasy", "ancient"]:
        if intro > 0.5:
            item = "小さな本や巻物を手に持っている"
        elif play > 0.5:
            item = "小さな光るものや、不思議な小物を指先でいじっている"
    elif world_type in ["future", "dystopia"]:
        if e > 0.5:
            item = "端末やカード状のデバイスを片手に持っている"
        else:
            item = "イヤホンやヘッドセットなど、通信機器がワンポイントになっている"
    else:
        if intro > 0.5:
            item = "マグカップやノートなど、日常的な小物をさりげなく持っている"

    if item:
        lines.append(f"- 小物：{item}")

    return "\n".join(lines)


# ========= 来世の恋人の似顔絵（イメージ） =========
def describe_next_life_partner_portrait(traits: dict, world_type: str, next_gender: str) -> str:
    """
    来世の恋人のビジュアルイメージ。
    world_type と next_gender に合わせて、ざっくりした似顔絵メモを返す。
    """

    sens = traits["sensitivity"]
    intro = traits["introspection"]
    e = traits["energy"]
    s = traits["stability"]

    lines = []
    lines.append("● 来世の恋人の似顔絵（イメージメモ）")

    # 恋人の性別
    if next_gender == "男性":
        # あなたが男性 → 恋人は女性
        base_face = "やわらかい目元で、表情の変化がわかりやすい女性"
    else:
        # あなたが女性 or その他 → 恋人は男性
        base_face = "落ち着いた目つきで、優しい印象のある男性"

    # 世界観ごとの雰囲気
    if world_type == "fantasy":
        bg = "少し光の粒が漂う森や、空の色が印象的な場所"
        outfit = "少しだけ装飾のある服や、自然な色合いの布を重ねた装い"
    elif world_type == "occult":
        bg = "室内の一角や、夜の街角など、少し静かな場所"
        outfit = "モノトーン寄りの服装に、さりげないアクセサリーが一つ"
    elif world_type == "dystopia":
        bg = "古びた建物や看板がある路地、少し無骨な街並み"
        outfit = "機能的なジャケットや、動きやすさ重視の服装"
    elif world_type == "future":
        bg = "シンプルな光のラインや、半透明のパネルがある近未来空間"
        outfit = "ミニマルで少しシャープなシルエットの服"
    elif world_type == "ancient":
        bg = "木々や石造りの建物、焚き火など、素朴な景色"
        outfit = "布をゆったりまとった衣装や、自然素材の服"
    else:
        bg = "日常の街角やカフェ、公園など、現実に近い場所"
        outfit = "シンプルで清潔感のあるカジュアルな服装"

    # 表情・距離感
    if sens > 0.6:
        face_detail = "目線が柔らかく、こちらの様子をよく見てくれている感じ"
    elif intro > 0.6:
        face_detail = "少し控えめな表情だが、目元に優しさがにじんでいる"
    else:
        face_detail = "自然体で、あまり身構えずに笑っている表情"

    if s > 0.7:
        pose = "姿勢が安定していて、横に立つと落ち着くような立ち方"
    elif e > 0.6:
        pose = "少し動きのあるポーズで、会話の途中を切り取ったような雰囲気"
    else:
        pose = "肩の力が抜けた、リラックスした立ち方"

    lines.append(f"- 顔立ちのイメージ：{base_face}")
    lines.append(f"- 背景：{bg}")
    lines.append(f"- 服装：{outfit}")
    lines.append(f"- 表情・ポーズ：{face_detail}、{pose}")

    return "\n".join(lines)


# ========= Streamlit UI =========
def main():

    # CSS記述のデザインの適用
    st.markdown(CUTE_CSS, unsafe_allow_html=True)




    st.set_page_config(
        page_title="声からわかる 来世の適職診断（細かめ分析版）",
        page_icon="🔮",
        layout="centered",
    )

    st.title("🔮 声からわかる『来世の適職診断』")
    st.caption(
        "性別と声の特徴から、あなたの“来世の雰囲気”をゆるく推定して、"
        "来世の適職・世界観・恋愛観などを診断するエンタメツールです。"
    )

    st.markdown("---")
    st.subheader("1. 性別と音声を入力")

    gender = st.selectbox("性別（ざっくりでOKです）", ["女性", "男性", "その他・ひみつ"])

    st.write(
        "普段どおりの話し方で、30秒〜1分ほど録音した音声ファイルをアップロードしてください。\n"
        "今日あったことや、最近考えていることなど、内容はなんでもOKです。"
    )

    uploaded = st.file_uploader(
        "対応形式: WAV / MP3 / OGG / M4A（環境による）",
        type=["wav", "mp3", "ogg", "m4a"],
    )

    if uploaded is None:
        st.info("性別を選び、音声ファイルをアップロードすると診断ボタンが表示されます。")
        return

    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(tmp_path)

    if st.button("来世の適職を診断する"):
        with st.spinner("あなたの声から、来世の世界とお仕事、生活や恋愛まで細かく読み取っています..."):
            try:
                features = extract_voice_features(tmp_path)
                # 来世の性別を推定
                traits = score_voice_traits(features)
                next_gender, gender_reason = infer_next_life_gender(traits, gender)
                world_scores = score_world_axes(traits)

                # 特性説明
                traits_text = describe_voice_traits(traits, world_scores)
                # プロフィール ＋ world_type
                profile_text, world_type = build_next_life_profile(traits, world_scores, next_gender)
                
                # 適職
                jobs = choose_next_life_jobs(traits, world_scores, world_type, next_gender)
                # アドバイス
                advice_text = advice_for_now(traits, world_scores)
                # 声
                next_voice_desc = describe_next_life_voice(traits, world_type)
                next_voice_sample_text = build_next_life_voice_sample_text(world_type, next_gender)
                # 来世の自分の似顔絵
                self_portrait_text = describe_next_life_self_portrait(traits, world_type, next_gender)
                # メッセージ
                message_text = build_message_from_next_life(traits, world_type, next_gender)
                # 恋愛
                love_text = describe_next_life_love_style(traits, world_type, next_gender)
                partner_text = describe_next_life_partner(traits, world_type, next_gender)
                # 来世の恋人の似顔絵
                partner_portrait_text = describe_next_life_partner_portrait(traits, world_type, next_gender)

                # TTS
                voice_audio_path = generate_next_life_voice_audio(next_voice_sample_text)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        st.markdown("---")
        st.subheader("2. あなたの声の特性（詳細）")
        st.text(traits_text)

        st.markdown("---")
        st.subheader("3. 来世のあなたの特徴（世界観＋人格）")
        st.text(profile_text)


        st.markdown("---")
        st.subheader("4. 来世の適職")

        st.markdown(f"### 🌟 メイン適職：**{jobs['main_job']}**")
        if jobs["sub_jobs"]:
            st.markdown("#### サブ候補")
            for j in jobs["sub_jobs"]:
                st.markdown(f"- {j}")

        st.markdown("#### なぜこの職業なのか？")
        st.write(jobs["explanation"])

        st.markdown("---")
        st.subheader("5. 来世のあなたはこんな声")

        st.text(next_voice_desc)

        st.markdown("#### 来世ボイス・サンプル")
        st.write("※テキストを合成したサンプル音声です。実際の声質そのものを再現するものではありません。")

        if voice_audio_path is not None:
            st.audio(voice_audio_path)
            st.caption("「来世のあなた」が今世のあなたに話しかけているイメージで再生してみてください。")
        else:
            if not GTTS_AVAILABLE:
                st.info("gTTS がインストールされていないため、音声再生はスキップされています。\n`pip install gTTS` で有効化できます。")
            else:
                st.info("環境の都合で音声生成に失敗したため、テキストのみの表示になります。")


        st.markdown("---")
        st.subheader("6. 来世のあなたの似顔絵")
        st.text(self_portrait_text)


        st.markdown("---")
        st.subheader("7. 来世のあなたから今世のあなたへのメッセージ")
        st.text(message_text)

        st.markdown("---")
        st.subheader("8. 来世の恋愛観")
        st.text(love_text)

        st.markdown("---")
        st.subheader("9. 来世のあなたの恋人")
        st.text(partner_text)

        st.markdown("---")
        st.subheader("10. 来世の恋人の似顔絵")
        st.text(partner_portrait_text)

        st.markdown("---")
        st.subheader("11. 来世のために、今できること")
        st.text(advice_text)

        st.markdown("---")
        st.caption(
            "※この診断はエンターテインメント目的です。"
            "　運命や将来を保証するものではありませんが、"
            "　“自分の声と対話するきっかけ”として楽しんでもらえたらうれしいです。"
        )


if __name__ == "__main__":
    main()
