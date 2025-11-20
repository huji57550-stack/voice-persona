# voice_animal_popularity.py
# 🐾 動物別 あなたのモテ度チェック
#
# 声の特徴から、犬・猫・アヒル・ゾウ・ペンギン・フクロウ・ライオン・イルカなど、
# 各動物界での「モテ度」を数値化して、ユーモラスに診断するエンタメツールです。

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

    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 無音対策：完全無音だと特徴量が壊れるので微小ノイズを足す
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


# ========= スコアリング =========
def normalize(v, vmin, vmax):
    if vmax - vmin == 0:
        return 0.5
    x = (v - vmin) / (vmax - vmin)
    return float(np.clip(x, 0.0, 1.0))


def score_voice_traits(features: dict) -> dict:
    """
    声の特徴からざっくり指標を作る：
      - energy      : 元気さ
      - tension     : 緊張感・とがり
      - stability   : 安定感
      - brightness  : 声の明るさ
      - roughness   : ザラつき
      - pitch_var   : 抑揚の大きさ
      - tempo_index : 話すテンポ
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

    return {
        "energy": energy,
        "tension": tension,
        "stability": stability,
        "brightness": brightness,
        "roughness": roughness,
        "pitch_var": pitch_var,
        "tempo_index": tempo_index,
    }


# ========= 動物別モテ度スコア =========
def compute_animal_scores(traits: dict) -> dict:
    """
    各動物が好みそうな声の特徴から「モテ度」を0〜100でスコア化（完全エンタメ）。
    """
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    scores = {}

    # 🐶 犬：明るくて元気で、でも安定感もある声が好き
    dog_pref = 0.4 * e + 0.2 * b + 0.3 * s + 0.1 * (1.0 - t)
    scores["犬"] = float(np.clip(dog_pref, 0.0, 1.0) * 100)

    # 🐱 猫：静かめで、安定していて、ちょっとクール寄りな声が好き
    cat_pref = 0.3 * (1.0 - e) + 0.3 * s + 0.2 * (1.0 - b) + 0.2 * (1.0 - ti)
    scores["猫"] = float(np.clip(cat_pref, 0.0, 1.0) * 100)

    # 🦆 アヒル：軽やかで明るくて、ちょっとザラっとした楽しい声が好き
    duck_pref = 0.3 * b + 0.3 * e + 0.2 * pv + 0.2 * r
    scores["アヒル"] = float(np.clip(duck_pref, 0.0, 1.0) * 100)

    # 🐘 ゾウ：低め・落ち着き・安定感。テンポもゆったり目が好き
    elephant_pref = 0.35 * s + 0.25 * (1.0 - b) + 0.2 * (1.0 - ti) + 0.2 * (1.0 - t)
    scores["ゾウ"] = float(np.clip(elephant_pref, 0.0, 1.0) * 100)

    # 🐧 ペンギン：ちょっと明るく、テンポもそこそこ、バランス型が好き
    penguin_pref = 0.25 * e + 0.25 * b + 0.25 * s + 0.25 * ti
    scores["ペンギン"] = float(np.clip(penguin_pref, 0.0, 1.0) * 100)

    # 🦉 フクロウ：夜っぽさ・低め・ゆっくり・観察者的な声が好き
    owl_pref = 0.4 * s + 0.2 * (1.0 - b) + 0.2 * (1.0 - ti) + 0.2 * (1.0 - e)
    scores["フクロウ"] = float(np.clip(owl_pref, 0.0, 1.0) * 100)

    # 🦁 ライオン：堂々・エネルギッシュ・ちょっと荒々しい声が好き
    lion_pref = 0.4 * e + 0.2 * r + 0.2 * b + 0.2 * (1.0 - t)
    scores["ライオン"] = float(np.clip(lion_pref, 0.0, 1.0) * 100)

    # 🐬 イルカ：明るくてよく動く、抑揚たっぷりの声が好き
    dolphin_pref = 0.3 * b + 0.25 * pv + 0.25 * ti + 0.2 * e
    scores["イルカ"] = float(np.clip(dolphin_pref, 0.0, 1.0) * 100)

    return scores


def make_animal_comment(name: str, score: float) -> str:
    """
    動物別に、スコアに応じたユーモラスなコメントを生成
    """
    s = score
    if name == "ゾウ":
        if s > 80:
            return (
                "どうしてあなたは人間に生まれてきてしまったのか……。\n"
                "ゾウ界に生まれていたら、橋本環奈バリにモテモテの人生だったのに！ 惜しい！"
            )
        elif s > 60:
            return (
                "ゾウ社会では“隣の群れからも評判聞こえてくるレベル”の人気者。\n"
                "草原を歩くだけで、耳パタパタさせながら振り向かれてます。"
            )
        elif s > 40:
            return (
                "ゾウ的には「なんか落ち着く声の人間だな〜」くらいの好印象。\n"
                "サファリパークで担当ゾウにだけ妙になつかれるタイプです。"
            )
        else:
            return (
                "ゾウ界ではそこまでモテモテではないけれど、\n"
                "たま〜に優しいおじいちゃんゾウに可愛がられるポジションです。"
            )

    if name == "アヒル":
        if s > 80:
            return (
                "アヒル池に立った瞬間、全アヒルが一斉にあなたの方を向くレベル。\n"
                "「クワッ（推し来た）」ってざわつきます。"
            )
        elif s > 60:
            return (
                "「あの人間の声、なんかテンション上がるクワ〜」と、\n"
                "アヒルたちの井戸端会議でちょこちょこ話題にされる人気っぷり。"
            )
        elif s > 40:
            return (
                "エサよりあなたの声をちょっと優先してくれるアヒルが数羽出るくらい。\n"
                "“じわモテ枠”として評価されています。"
            )
        else:
            return (
                "アヒル的には「悪くないけど、エサの方が大事クワ」な立ち位置。\n"
                "ただし、通ううちに“常連の人間”枠でじわじわ好感度アップの余地あり。"
            )

    if name == "犬":
        if s > 80:
            return (
                "ドッグランに一歩入った瞬間、全犬が尻尾ブンブンで寄ってくるレベル。\n"
                "人間界でのモテより、犬界でのモテ度の方が明らかに高いです。"
            )
        elif s > 60:
            return (
                "「この声の人間、散歩うまそう…！」と、\n"
                "勝手に心のリードを渡されるくらいには信頼されています。"
            )
        elif s > 40:
            return (
                "しっぽは振らないけど、耳だけピクッと反応されるくらいの好印象。\n"
                "“友だち以上、飼い主未満”ポジです。"
            )
        else:
            return (
                "犬的には「嫌いではないが、そこまでテンションは上がらない」くらい。\n"
                "でも、おやつを持った瞬間、一瞬で評価がひっくり返る余地あり。"
            )

    if name == "猫":
        if s > 80:
            return (
                "猫カフェに入った瞬間、なぜか全員こっそり距離を詰めてくるレベル。\n"
                "「いや別に好きとかじゃないし…」と言いながら、隣で丸くなるやつです。"
            )
        elif s > 60:
            return (
                "猫たちの心のメモに「この人間＝居心地いい」と登録されるくらいの好感度。\n"
                "スマホいじってるだけで、いつの間にか膝を奪われます。"
            )
        elif s > 40:
            return (
                "猫的には「悪くないけど、撫で方次第かな〜」ぐらいの微妙なライン。\n"
                "静かめに話すと、評価が1段階アップしそうです。"
            )
        else:
            return (
                "猫界ではまだ“通りすがりの人間”ランクですが、\n"
                "気が向いた猫がたまにスリスリしに来る、隠れファンはいます。"
            )

    if name == "ペンギン":
        if s > 80:
            return (
                "南極であなたが一声発した瞬間、ペンギンたちの大行列ができるレベル。\n"
                "求愛ダンスの相手に指名されまくります。"
            )
        elif s > 60:
            return (
                "ペンギン的には「この人間のそば、なんか温かい気がする」と大人気。\n"
                "写真を撮ろうとすると、絶妙な位置取りでフレームに入ってきます。"
            )
        elif s > 40:
            return (
                "ペンギンたちから“ご近所さん”として、\n"
                "軽く会釈してもらえるくらいの距離感。悪くないです。"
            )
        else:
            return (
                "ペンギン界では今のところ“観光客”扱いですが、\n"
                "通い続ければ、そのうち一羽くらいはあなたを覚えてくれます。"
            )

    if name == "フクロウ":
        if s > 80:
            return (
                "深夜の森であなたが話し出すと、フクロウたちの会議が中断するレベル。\n"
                "“賢者ポジの人間”として議事録に残されます。"
            )
        elif s > 60:
            return (
                "フクロウ的には「この声、落ち着いて観察に集中できる」と高評価。\n"
                "あなたの近くの木に、なぜかいつも同じフクロウがいます。"
            )
        elif s > 40:
            return (
                "夜道を歩いていると、電線の上からじっと見られているかもしれません。\n"
                "それ、ちょっとあなたの声が気になってるフクロウです。"
            )
        else:
            return (
                "フクロウにとってはまだ“背景音”くらいの存在ですが、\n"
                "静かなトーンでゆっくり話すと、少しずつ記憶に残り始めます。"
            )

    if name == "ライオン":
        if s > 80:
            return (
                "サバンナであなたが叫ぶと、ライオンたちが\n"
                "「あいつ、ボスじゃね？」とザワつくレベルのカリスマ声。"
            )
        elif s > 60:
            return (
                "ライオン的には「一緒に狩りに行くならこの人間かな」くらいの信頼度。\n"
                "群れの中で“副リーダー”ポジに抜擢されるタイプです。"
            )
        elif s > 40:
            return (
                "遠くで吠えてるライオンが、たまにあなたの声に合わせて吠え返してくれるかも。\n"
                "“悪くない共鳴”くらいには認識されています。"
            )
        else:
            return (
                "ライオンからすると「敵ではないが、特に獲物でもない」絶妙ポジション。\n"
                "でも、安全に近くで観察してもらえると思えば、むしろお得です。"
            )

    if name == "イルカ":
        if s > 80:
            return (
                "水族館であなたが話すと、イルカたちが\n"
                "「ちょっとその声でもう一回喋って！」とばかりにジャンプ連発。"
            )
        elif s > 60:
            return (
                "イルカ的には「この人間と一緒に泳いだら絶対楽しい」と確信しているレベル。\n"
                "ショーの主役に指名されるのも時間の問題です。"
            )
        elif s > 40:
            return (
                "イルカたちの中で“ちょっと気になる人間”枠。\n"
                "ガラス越しに、たまに目を合わせに来てくれます。"
            )
        else:
            return (
                "今のところイルカ界では“謎の人間”ですが、\n"
                "高め・明るめのトーンで話すと、じわじわ人気が出てきます。"
            )

    # デフォルト（ここには来ない想定）
    return "この動物界でも、そこそこ良いポジションを取れそうな気配があります。"


def make_overall_story(top_name: str, top_score: float) -> str:
    """
    一番モテ度の高い動物に合わせて、まとめの一言ストーリーを作る
    """
    s = top_score
    if top_name == "ゾウ" and s > 70:
        return (
            "総合コメント：\n"
            "どうしてあなたは人間に生まれてきてしまったのか！\n"
            "ゾウに生まれていたら、橋本環奈バリにモテモテの人生だったのに！ 惜しい！！"
        )
    if top_name == "犬":
        return (
            "総合コメント：\n"
            "人間界ではそこそこでも、犬界では“伝説の飼い主候補”クラス。\n"
            "前世、もしくは来世は犬のリーダーかもしれません。"
        )
    if top_name == "猫":
        return (
            "総合コメント：\n"
            "人間からのモテを目指してきたかもしれませんが、\n"
            "本当のフィールドは“ツンデレ猫の心を溶かす声”だったようです。"
        )
    if top_name == "アヒル":
        return (
            "総合コメント：\n"
            "人間社会では伝わりきっていないかもしれない、この軽やかさと陽気さ。\n"
            "アヒル界では“池のセンター”を任されるレベルのスター素質です。"
        )
    if top_name == "ペンギン":
        return (
            "総合コメント：\n"
            "実はあなた、南極に転生した方が恋愛運が爆上がりするタイプかも。\n"
            "ペンギンたちの間で、あなたの声は“求愛ダンス確定BGM”です。"
        )
    if top_name == "フクロウ":
        return (
            "総合コメント：\n"
            "昼間よりも、夜の静かな時間帯に本領発揮する声。\n"
            "フクロウたちからは“相談したくなる声の人間”として密かに憧れられています。"
        )
    if top_name == "ライオン":
        return (
            "総合コメント：\n"
            "人間界では“ちょっと頼れそうな人”くらいの評価かもしれませんが、\n"
            "ライオン界では「王族候補」の声質です。サバンナでの人生、ワンチャンあり。"
        )
    if top_name == "イルカ":
        return (
            "総合コメント：\n"
            "イルカショーのMCよりも、あなたがしゃべった方が盛り上がる可能性大。\n"
            "海に落ちても、イルカたちが全力で助けに来てくれるタイプです。"
        )

    return (
        "総合コメント：\n"
        "人間界だけで勝負しているのがもったいないレベルで、動物界でのポテンシャルが高めです。\n"
        "来世ガチャでは、ぜひ今のトップ動物を狙ってみてください。"
    )


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
    lines.append("※これはあくまで簡易な音声特徴から遊ぶエンタメ診断です。深く考えすぎずネタとしてどうぞ。")
    return "\n".join(lines)


# ========= Streamlit UI =========
def main():


    # CSS記述のデザインの適用
    st.markdown(CUTE_CSS, unsafe_allow_html=True)


    st.set_page_config(page_title="動物別あなたのモテ度チェック", page_icon="🐾", layout="centered")

    st.title("🐾 動物別 あなたのモテ度チェック")
    st.caption(
        "あなたの声は、犬・猫・アヒル・ゾウ・ペンギン・フクロウ・ライオン・イルカなど、"
        "どの動物界でいちばんモテるのか？をゆるく診断するエンタメツールです。"
    )

    st.markdown("---")
    st.subheader("1. 声の録音")

    st.write(
        "普段どおりの話し方で、30秒〜1分ほど話した音声がおすすめです。\n"
        "最近の出来事や、好きなものについて話して録音してみてください。"
    )

    uploaded = st.audio_input("🎤 マイクで声を録音してください")

    if uploaded is None:
        st.info("マイクで録音すると、診断ボタンが表示されます。")
        return

    # 一時ファイルに保存
    audio_bytes = uploaded.getbuffer()
    suffix = ".wav"   # 録音形式は WAV 固定
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(tmp_path)

    if st.button("動物別モテ度を診断する"):
        with st.spinner("動物たちの“推し声”ランキングを集計中..."):
            try:
                features = extract_voice_features(tmp_path)
                traits = score_voice_traits(features)
                scores = compute_animal_scores(traits)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        # スコアでソート
        sorted_animals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_name, top_score = sorted_animals[0]

        st.markdown("---")
        st.subheader("2. 動物界トータルランキング（トップ3）")

        for i, (name, score) in enumerate(sorted_animals[:3], start=1):
            st.markdown(f"**{i}位：{name}界モテ度　→　{score:.1f} / 100**")
            st.write(make_animal_comment(name, score))
            st.markdown("")

        st.markdown("---")
        st.subheader("3. あなたが一番モテる動物界まとめ")

        story = make_overall_story(top_name, top_score)
        st.write(story)

        st.markdown("---")
        st.subheader("4. 声の特徴レポート")
        st.text(explain_traits(traits))

        st.markdown("---")
        st.caption(
            "※この診断はジョーク用です。実際の動物との相性や動物行動学とは一切関係ありません。\n"
            "　「人間に生まれてきちゃったけど、別の世界線ではモテモテだったかも…」くらいのノリで楽しんでください。"
        )


if __name__ == "__main__":
    main()
