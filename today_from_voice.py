# today_from_voice.py
# 🎙 声でわかるあなたの今日（詳細分析フルバージョン）

import tempfile
from pathlib import Path
import random

import numpy as np
import librosa
import streamlit as st


# ========= 音声特徴量の抽出 =========
def extract_voice_features(file_path: str) -> dict:
    """音声ファイルから簡易特徴量を抽出してスコア化する"""

    # sr=None で元のサンプリングレートのまま読み込み
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 無音対策（小さなノイズを足す）
    if np.max(np.abs(y)) < 1e-4:
        y = y + np.random.normal(0, 1e-4, size=len(y))

    # 基本的な特徴量
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]  # 大きさ（音量）
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]  # 雑さ・ノイズ感
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # 明るさ
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # テンポ（ざっくり）

    # ピッチ推定（平均とばらつき）
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_valid = f0[np.isfinite(f0)]
    if len(f0_valid) == 0:
        f0_mean = 0.0
        f0_std = 0.0
    else:
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))

    features = {
        "duration": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "tempo": float(tempo),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
    }
    return features


# ========= スコアリング =========
def normalize(value, vmin, vmax):
    """簡易正規化（0〜1）"""
    if vmax - vmin == 0:
        return 0.5
    v = (value - vmin) / (vmax - vmin)
    return float(np.clip(v, 0.0, 1.0))


def score_from_features(features: dict) -> dict:
    """
    特徴量から 0〜1 のスコアを作る
    - energy: 元気さ・外向きモード
    - tension: 緊張・力み
    - stability: 声の安定感・落ち着き
    - brightness: 声の明るさ（高音寄りかどうか）
    - roughness: ザラつき・ノイズ感
    - pitch_var: 声の抑揚（高さのブレ）
    - tempo_index: テンポ感（早口寄りかどうか）
    """
    rms_mean = features["rms_mean"]
    rms_std = features["rms_std"]
    zcr_mean = features["zcr_mean"]
    centroid_mean = features["centroid_mean"]
    tempo = features["tempo"]
    f0_std = features["f0_std"]

    # メイン3軸
    energy = 0.6 * normalize(rms_mean, 0.005, 0.05) + 0.4 * normalize(tempo, 60, 180)
    tension = (
        0.4 * normalize(zcr_mean, 0.01, 0.2)
        + 0.3 * normalize(centroid_mean, 1000, 4000)
        + 0.3 * normalize(rms_std, 0.0, 0.03)
    )
    stability = 1.0 - 0.5 * normalize(f0_std, 0, 40) - 0.5 * normalize(rms_std, 0.0, 0.03)
    stability = float(np.clip(stability, 0.0, 1.0))

    # 追加の説明用指標
    brightness = normalize(centroid_mean, 1000, 4000)  # 明るめ・高音寄りか
    roughness = normalize(zcr_mean, 0.01, 0.2)        # ザラつき・ノイズ感
    pitch_var = normalize(f0_std, 0, 40)              # 声の抑揚の大きさ
    tempo_index = normalize(tempo, 60, 180)           # テンポの速さ

    return {
        "energy": energy,
        "tension": tension,
        "stability": stability,
        "brightness": brightness,
        "roughness": roughness,
        "pitch_var": pitch_var,
        "tempo_index": tempo_index,
    }


# ========= コンディション詳細解説 =========
def describe_condition(scores: dict) -> str:
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]
    b = scores["brightness"]
    r = scores["roughness"]
    pv = scores["pitch_var"]
    ti = scores["tempo_index"]

    # --- ベースのコンディション ---
    if e > 0.7 and t < 0.6:
        mood = (
            "エネルギーがしっかり乗っていて、かつ暴走しすぎていない、“攻めやすい”コンディションです。"
            "外向きの予定や、新しいチャレンジとの相性が良い日。"
        )
    elif e > 0.7 and t >= 0.6:
        mood = (
            "テンション自体は高めで、アクセルは踏めている状態です。ただし、やや緊張や力みも同時に入っていて、"
            "発言が強く出やすい・早口になりやすいなどの“空回りリスク”も少しあります。"
        )
    elif e < 0.4 and s > 0.6:
        mood = (
            "エネルギーは控えめですが、声の揺れは少なく、落ち着いた安定したコンディションです。"
            "がつがつ行く日というより、“腰を据えてじっくり取り組むタスク”と相性が良さそう。"
        )
    elif e < 0.4 and s < 0.5:
        mood = (
            "少しお疲れモード、あるいは集中しきれない感じが声ににじんでいるかもしれません。"
            "頑張れば動けるけれど、無理を続けると反動が出やすい日です。"
        )
    else:
        mood = (
            "エネルギー・緊張感・安定感のどれか一つが突出しているわけではない、ニュートラル寄りの状態です。"
            "“やろうと思えばなんでもできる”代わりに、“流されるとなんとなく一日が終わる”パターンにもなりやすい日。"
        )

    # --- 一言コメント + 今日のミニアクション ---
    if t > 0.7:
        one_liner = (
            "少し緊張や力みが乗りやすい声になっています。"
            "今日のミニアクション：階段を上るときに、1段目だけ“かかとまでしっかり床につけてから”上る・"
            "人と話す前に首と肩を一回ぐるっと回す、など“動きをゆっくりにする癖”をどこかに入れてみて。"
        )
    elif s > 0.7:
        one_liner = (
            "声は全体的に安定していて、聞き手に安心感を与えやすい状態です。"
            "今日のミニアクション：話しかけられたときに、まず“口角だけ2mm上げてから返事をする”ことを意識してみて。"
            "それだけで、相手からの受け取られ方が一段やわらかくなります。"
        )
    elif e > 0.7:
        one_liner = (
            "エネルギーがしっかり乗っていて、前に出ていきやすい声です。"
            "今日のミニアクション：階段やエスカレーターで“最初の1段だけ、少し大股で踏み出す”イメージで上ってみて。"
            "歩くときも、最初の3歩だけ歩幅を少し広げると、自然と気持ちも前向きになります。"
        )
    elif e < 0.4:
        one_liner = (
            "少しペースを落としたい声の状態。ムリにテンションを上げなくても大丈夫な日です。"
            "今日のミニアクション：移動のたびに“立ち止まって背伸びを一回する”／"
            "コンビニやレジ待ちで“足の裏全体を床にぎゅっと押しつける感覚”を味わってみて。"
            "小さく体をほどくと、頭もゆるみやすくなります。"
        )
    else:
        one_liner = (
            "コンディションはおおむね平均的。ちょっとの一工夫で、どちら側にも振れる状態です。"
            "今日のミニアクション：エスカレーターで立っている時間に“深呼吸を3回する”／"
            "エレベーターを待つ間だけ“背筋をすっと伸ばす”など、生活動作に1つだけ習慣を差し込んでみて。"
        )

    # --- 声質の傾向コメント ---
    traits = []

    # 明るさ
    if b > 0.7:
        traits.append("・声のトーンはやや明るめ〜高め寄りで、元気さやカジュアルさが出やすい状態です。")
    elif b < 0.3:
        traits.append("・声のトーンはやや低め〜落ち着き寄りで、信頼感や安心感が出やすい状態です。")
    else:
        traits.append("・声の高さは中庸で、ビジネス／雑談どちらにも振りやすいレンジです。")

    # ザラつき
    if r > 0.7:
        traits.append("・少しザラつき・ノイズ感が強めで、“勢い”や“粗さ”を感じさせる可能性があります。（マイク環境の影響もあり）")
    elif r < 0.3:
        traits.append("・ノイズやザラつきは少なく、クリアで聞き取りやすい声質になっています。")

    # 抑揚
    if pv > 0.7:
        traits.append("・声の高さの変化が大きく、“感情表現豊か・リアクション大きめ”に聞こえやすい傾向があります。")
    elif pv < 0.3:
        traits.append("・声の高さの変化は少なめで、“落ち着いた・フラットな印象”を与えやすい状態です。")

    # テンポ
    if ti > 0.7:
        traits.append("・全体のテンポはやや速め寄り。情報量が多い話や説明では、意識的に“間”を入れると伝わりやすくなります。")
    elif ti < 0.3:
        traits.append("・テンポはゆったりめ。聞き手に安心感は与えやすい一方で、重要なポイントだけ少しテンポを上げると締まりが出ます。")

    traits_text = "\n".join(traits) if traits else "・大きな偏りはなく、バランス型の声質です。"

    # --- 数値解説（※「今日の運用ポイント」はナシ）---
    detail = (
        f"\n\n● スコア概要（0〜1）\n"
        f"- 元気さ（energy）     : {e:.2f}\n"
        f"- 緊張感（tension）    : {t:.2f}\n"
        f"- 安定感（stability）  : {s:.2f}\n"
        f"- 明るさ（brightness） : {b:.2f}\n"
        f"- ザラつき（roughness）: {r:.2f}\n"
        f"- 抑揚（pitch_var）     : {pv:.2f}\n"
        f"- テンポ（tempo_index）: {ti:.2f}\n"
        "\n● 声質のざっくり傾向\n"
        f"{traits_text}\n"
    )

    return f"● 今日の声コンディション\n{mood}\n\n● 一言コメント\n{one_liner}{detail}"


# ========= 行動ごとの“やる・やめる”診断 =========
def judge_activity(scores: dict) -> dict:
    """
    行動ごとの「おすすめ度」とコメント
    """
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]

    def label_from_score(score):
        if score >= 0.7:
            return "◎ とてもおすすめ（今日がチャンス）"
        elif score >= 0.4:
            return "◯ 工夫すればおすすめ（日程調整次第）"
        else:
            return "△ 今日は無理せず、整える日にしてもOK"

    result = {}

    # --- プレゼン・商談 ---
    score_prez = 0.5 * e + 0.3 * s - 0.2 * t
    prez_advice = []

    if score_prez >= 0.7:
        prez_advice.append("・声のエネルギーと安定感のバランスがよく、説得力を出しやすい日です。")
        prez_advice.append("・話し始めの30秒〜1分を“ゆっくり・低めトーン”にすることで、さらに信頼感が増します。")
        prez_advice.append("・資料の読み上げよりも、自分の言葉で補足説明を入れると、熱量が伝わりやすくなります。")
    elif score_prez >= 0.4:
        prez_advice.append("・コンディションはまずまず。準備を丁寧にすれば十分戦えます。")
        prez_advice.append("・本番前に“重要なフレーズだけを声に出して練習”しておくと、滑舌と安心感がアップします。")
        prez_advice.append("・想定質問を2〜3個だけ書き出しておくと、緊張による沈黙を防ぎやすくなります。")
    else:
        prez_advice.append("・今日が本命プレゼンなら、可能なら日程調整も検討しても良いレベル。")
        prez_advice.append("・どうしても外せない場合は、時間配分と“話さない部分（削る部分）”を明確にして臨みましょう。")
        prez_advice.append("・開始直前に、深呼吸×3回＋ゆっくり水を一口飲むだけでも、声の震えがかなり違ってきます。")

    result["プレゼン・商談"] = {
        "score": float(np.clip(score_prez, 0.0, 1.0)),
        "label": label_from_score(score_prez),
        "advice": "\n".join(prez_advice),
    }

    # --- 飲み会・交流 ---
    score_party = 0.6 * e - 0.2 * t + 0.2 * s
    party_advice = []

    if score_party >= 0.7:
        party_advice.append("・会話のリード役、場をあたためるポジションに回るとハマりやすい日です。")
        party_advice.append("・最初の30分は“聞き役6：話し役4”くらいにすると、好感度がさらに上がります。")
        party_advice.append("・盛り上がってきたら、1回だけ“真面目な話”を差し込むと、印象に残りやすくなります。")
    elif score_party >= 0.4:
        party_advice.append("・無理して盛り上げ役にならなくてもOK。少人数でじっくり話す席だと居心地が良さそう。")
        party_advice.append("・一人ひとりの話を少し深掘りする質問（なぜ？どうしてそう思った？）を投げると、会話の質が上がります。")
    else:
        party_advice.append("・今日は“行くなら短時間参加”くらいの気持ちでちょうどいいかもしれません。")
        party_advice.append("・無理にテンションを合わせるより、“聞き上手ポジション”に徹した方が疲れず、印象も◎。")

    result["飲み会・交流"] = {
        "score": float(np.clip(score_party, 0.0, 1.0)),
        "label": label_from_score(score_party),
        "advice": "\n".join(party_advice),
    }

    # --- 引っ越し・大きな作業 ---
    score_move = 0.5 * e + 0.5 * s
    move_advice = []

    if score_move >= 0.7:
        move_advice.append("・体力もメンタルもそこそこ整っているので、“一気に片付ける日”として向いています。")
        move_advice.append("・午前：重い作業／午後：軽い作業、のように強弱をつけてタスクを並べると◎。")
        move_advice.append("・30〜60分に1回は“5分だけ座る・水を飲む”を意識すると、疲れが翌日に残りにくくなります。")
    elif score_move >= 0.4:
        move_advice.append("・全部やり切ろうとせず、“今日はここまで”と範囲を決めてから着手すると良さそう。")
        move_advice.append("・最初の15分だけタイマーをかけて、ウォームアップ的に動いてみるのもおすすめです。")
    else:
        move_advice.append("・今日が本番引っ越し日でないなら、作業計画を立てる日・不要なものリストを作る日に回してもOK。")
        move_advice.append("・どうしてもやる場合は、“誰か1人手伝いを呼ぶ”だけでも、心理的負荷がかなり軽くなります。")

    result["引っ越し・大きな作業"] = {
        "score": float(np.clip(score_move, 0.0, 1.0)),
        "label": label_from_score(score_move),
        "advice": "\n".join(move_advice),
    }

    # --- テスト・試験 ---
    score_exam = 0.4 * s + 0.3 * (1 - t) + 0.3 * (1 - abs(e - 0.5))
    exam_advice = []

    if score_exam >= 0.7:
        exam_advice.append("・落ち着きと適度な集中状態が両立している、かなり試験向きのコンディションです。")
        exam_advice.append("・本番前は新しい知識を詰め込むより、“解ける問題を1〜2問確認する”くらいに留めると、安心感が維持されます。")
    elif score_exam >= 0.4:
        exam_advice.append("・コンディションは悪くありませんが、緊張や眠気などに振られる可能性もある日です。")
        exam_advice.append("・開始1〜2時間前に“軽く体を動かす（ストレッチ・散歩）”ことで、頭がクリアになります。")
    else:
        exam_advice.append("・体調・メンタル的にはやや不安定さもありそうな状態です。")
        exam_advice.append("・復習範囲を欲張りすぎず、“絶対に落としたくないテーマだけを確認する”と割り切るのがおすすめです。")

    result["テスト・試験"] = {
        "score": float(np.clip(score_exam, 0.0, 1.0)),
        "label": label_from_score(score_exam),
        "advice": "\n".join(exam_advice),
    }

    # --- 告白・大事な話 ---
    score_confess = 0.4 * e + 0.4 * s - 0.2 * t
    confess_advice = []

    if score_confess >= 0.7:
        confess_advice.append("・気持ちと声の乗り方がちょうどよく、“ストレートに伝える”のに向いている日です。")
        confess_advice.append("・言葉はシンプルに、伝えたい内容を“2〜3行だけメモ”しておくと、緊張しても迷いにくくなります。")
        confess_advice.append("・事前に、深呼吸しながら1回だけ声に出して練習しておくと、本番の揺れが減ります。")
    elif score_confess >= 0.4:
        confess_advice.append("・十分トライして良い日ですが、タイミングや時間帯を工夫すると成功率が上がりそうです。")
        confess_advice.append("・まずは“軽めの相談や雑談”から入り、相手がリラックスしているタイミングを見極めてから本題へ。")
    else:
        confess_advice.append("・今日がどうしても外せない日でなければ、“距離を少し縮める会話”にとどめる選択肢もありです。")
        confess_advice.append("・伝えるとしても、時間は短く・言葉はシンプルに。“また改めて話させて”と余白を残すのも手です。")

    result["告白・大事な話"] = {
        "score": float(np.clip(score_confess, 0.0, 1.0)),
        "label": label_from_score(score_confess),
        "advice": "\n".join(confess_advice),
    }

    return result


# ========= ラッキーアイテム =========
def pick_lucky_item(scores: dict) -> dict:
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]

    if e > 0.7:
        items = [
            ("炭酸飲料", "シュワっとした刺激でテンションを良い方向にキープ。"),
            ("赤い小物", "情熱と自信を後押ししてくれるカラー。"),
            ("ミント系タブレット", "頭をシャキっとさせてくれるスイッチアイテム。"),
        ]
    elif s > 0.7:
        items = [
            ("あたたかいお茶", "落ち着いた状態をそのままキープしてくれます。"),
            ("青いペン", "冷静な判断やロジカルな思考をサポート。"),
            ("ノート", "頭の中を整理すると、今日はさらに冴えます。"),
        ]
    elif t > 0.7:
        items = [
            ("ハンドクリーム", "ふと手元に意識を向けると、緊張が和らぎます。"),
            ("深呼吸アプリ", "1分だけでも呼吸を整える時間を。"),
            ("お気に入りの香り", "嗅覚からリラックスモードに切り替え。"),
        ]
    else:
        items = [
            ("イヤホン", "好きな音楽でコンディションを微調整できる万能アイテム。"),
            ("メモ帳", "ひらめきをすぐ書き留めると良い流れが続きます。"),
            ("飴・チョコレート", "少し糖分を入れて、集中力をプラス。"),
        ]

    item, reason = random.choice(items)
    return {"item": item, "reason": reason}


# ========= ファッション =========
def suggest_fashion(scores: dict) -> dict:
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]

    # ---- エネルギー高め × 緊張低め ----
    if e > 0.7 and t < 0.6:
        title = "今日は“赤＋猫モチーフ”が相性のいい日"
        detail = (
            "・どこか一か所に、赤を取り入れてみてください（靴下・小物・インナーなど一部でOK）。\n"
            "・アクセサリーや小物は、猫モチーフのものを一つだけ身につけると、今日の前向きさとよく馴染みます。"
        )

    # ---- エネルギー低め × 安定感高め ----
    elif e < 0.4 and s > 0.6:
        title = "今日は“青＋月モチーフ”が心に合う日"
        detail = (
            "・どこか一か所に、青を取り入れてみてください（バッグ・タオル・ペンなど、さりげないものでOK）。\n"
            "・アクセサリーや小物は、月モチーフのものを一つだけ身につけると、落ち着いたペースと相性が良いです。"
        )

    # ---- 緊張が強い日 ----
    elif t > 0.7:
        title = "今日は“水色＋小さな鳥モチーフ”が安心感をつくる日"
        detail = (
            "・どこか一か所に、淡い水色を入れてください（ノート・ハンカチ・スマホケースなどでもOK）。\n"
            "・アクセサリーや小物は、小さな鳥モチーフのものを一つだけ取り入れると、緊張がやわらぎやすい日です。"
        )

    # ---- ニュートラルな日 ----
    else:
        title = "今日は“白＋シンプルな腕時計”で整える日"
        detail = (
            "・どこか一か所に、白を取り入れてください（インナー・靴・ハンカチなど）。\n"
            "・アクセサリーや小物は、シンプルな腕時計を一つだけ選ぶと、全体がすっきり整って見える日です。"
        )

    return {"title": title, "detail": detail}


# ========= ランチ =========

def suggest_lunch(scores: dict) -> dict:
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]

    # ---- エネルギー高めの日 ----
    if e > 0.7:
        menu = "チキンソテー"
        title = f" **{menu}**"
        detail = (
            "今日は、外向きのエネルギーがしっかり乗っている一日。\n"
            "チキンは「地に足のついた行動力」を象徴するフードと言われることがあります。\n"
            "シンプルに焼いたチキンは、余分な装飾のない“まっすぐな決断力”をサポートしてくれる組み合わせ。\n"
            "午前中に溜めた気合いを、午後もほどよくキープしたい日のエネルギーチャージにぴったりです。"
        )

    # ---- エネルギー低め × 安定感高めの日 ----
    elif e < 0.4 and s > 0.6:
        menu = "ささみの梅しそ巻き"
        title = f" **{menu}**"
        detail = (
            "今日は、静かに自分のペースを守りやすい“調整モード”の日。\n"
            "ささみの軽さは、身体と心をふわっと軽くする“風のエレメント”のような存在。\n"
            "そこに梅としそという浄化感のある組み合わせが入ることで、\n"
            "いらない緊張やモヤモヤを、そっと手放していくのを後押ししてくれます。"
        )

    # ---- 緊張が強めの日 ----
    elif t > 0.7:
        menu = "豆腐ハンバーグ"
        title = f" **{menu}**"
        detail = (
            "今日は、少し心と身体が張りつめやすいコンディション。\n"
            "豆腐ベースのやわらかいハンバーグは、“土のエレメント”のように真ん中に戻してくれる存在です。\n"
            "しっかりしているのに重すぎない質感が、「がんばりすぎ」と「ゆるみすぎ」のちょうど間をつくってくれます。\n"
            "午後も落ち着いた集中を保ちたいときの、心をほっとさせる一皿です。"
        )

    # ---- 平均〜バランス日 ----
    else:
        menu = "サーモンソテー"
        title = f" **{menu}**"
        detail = (
            "今日はどちらかというと“整える一日”。\n"
            "サーモンは、水のエレメントと相性が良い食材と言われることがあります。\n"
            "流れに合わせてしなやかに進んでいく鮭のイメージは、「無理に頑張りすぎず、それでも前に進む力」。\n"
            "そこにバターのあたたかなコクが加わることで、\n"
            "静かなまま、でもちゃんとエネルギーをチャージしていきたい日のお守りランチになります。"
        )

    return {"title": title, "detail": detail}


# ========= おやつ =========

def suggest_snack(scores: dict) -> dict:
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]

    # ---- エネルギー高めの日 ----
    if e > 0.7:
        menu = "チョコレート"
        title = f" **{menu}**"
        detail = (
            "今日は前に進む力がしっかりある日。\n"
            "少しのチョコレートは、“もうひと押ししたい自分”をそっと後押ししてくれる、小さなブーストになります。"
        )

    # ---- エネルギー低め × 落ち着いている日 ----
    elif e < 0.4 and s > 0.6:
        menu = "ヨーグルト"
        title = f" **{menu}**"
        detail = (
            "今日は静かに整えていく流れが心地いい日。\n"
            "ヨーグルトのやわらかさは、“がんばりを足す”よりも“自分をいたわる”方向に気持ちをそっと戻してくれます。"
        )

    # ---- 緊張が強めの日 ----
    elif t > 0.7:
        menu = "バタークッキー"
        title = f" **{menu}**"
        detail = (
            "今日は少し心が張りつめやすいコンディション。\n"
            "バタークッキーのほろっとした甘さは、“力を抜いても大丈夫だよ”と教えてくれる、安心感のスイッチになります。"
        )

    # ---- 平均〜バランスの日 ----
    else:
        menu = "ナッツと小さめのチョコ"
        title = f" **{menu}**"
        detail = (
            "今日はどちらにも振れるバランスモードの日。\n"
            "ナッツと小さめのチョコの組み合わせは、“ちょっとだけ自分を満たす”ことで、一日のリズムを穏やかに整えてくれます。"
        )

    return {"title": title, "detail": detail}



# ========= 過ごし方 =========

def suggest_day_plan(scores: dict) -> dict:
    e = scores["energy"]
    t = scores["tension"]
    s = scores["stability"]

    # ---- エネルギー高めの日（やる気・行動力が乗る） ----
    if e > 0.7:
        plan = "“気になっていたことを1つだけ片づける日”"
        title = f"今日の過ごし方は **{plan}**"
        detail = (
            "エネルギーがしっかりある日は、行動を1つだけ前に進めるのが最も効果的です。\n"
            "・買おうか迷っていた日用品を1つ買う\n"
            "・溜めていた連絡を1件返す\n"
            "・部屋の一角だけ5分だけ片づける\n"
            "など、ハードルは低めでOK。\n"
            "“動き始めの1つ”が、今日のあなたのリズムを一気に整えてくれます。"
        )

    # ---- エネルギー低め × 安定していて落ち着いている日 ----
    elif e < 0.4 and s > 0.6:
        plan = "“ゆるめのルーティンを優先する日”"
        title = f"今日の過ごし方は **{plan}**"
        detail = (
            "落ち着いたコンディションの日は、新しいことより“整えること”が心地よくハマります。\n"
            "・お気に入りのカフェラテを飲む\n"
            "・散歩を10分だけする\n"
            "・机を軽く拭いてリセットする\n"
            "・湯船に3分だけ浸かる\n"
            "など、ゆるい行動がちょうどいいバランスに。\n"
            "無理にテンションを上げなくても、自然と気持ちが整っていきます。"
        )

    # ---- 緊張が強い日（気持ちが張っている・ソワソワしやすい） ----
    elif t > 0.7:
        plan = "“丁寧な休憩を小まめに入れる日”"
        title = f"今日の過ごし方は **{plan}**"
        detail = (
            "少し緊張が出やすい日は、がんばるより“こまめにほぐす”が正解です。\n"
            "・飲み物をゆっくり一口だけ飲む\n"
            "・深呼吸を3回する\n"
            "・スマホを1分伏せる\n"
            "などのミニ休憩をはさむと、午後の自分が驚くほどラクになります。\n"
            "予定がある人は、“始まる直前より数分前に軽く休む”のが特におすすめ。"
        )

    # ---- 平均〜バランスの日（流れで良い日） ----
    else:
        plan = "“好きなことを15分だけやる日”"
        title = f"今日の過ごし方は **{plan}**"
        detail = (
            "特に偏りのない日だからこそ、“好きなことを少しだけ”が心の栄養になります。\n"
            "・好きな音楽を1曲だけ聴く\n"
            "・お気に入りのSNSを15分だけ見る\n"
            "・読みかけの本を数ページ進める\n"
            "・動画を1本見る\n"
            "など、楽しいことを軽く取り入れるだけで、今日の満足度がぐっと上がります。\n"
            "気ままに過ごしてOKな日です。"
        )

    return {"title": title, "detail": detail}



# ========= Streamlit UI =========
def main():
    st.set_page_config(page_title="声でわかるあなたの今日", page_icon="🎙", layout="centered")

    st.title("🎙 声でわかるあなたの今日")
    st.caption("声のざっくり特徴から、今日のコンディションと “やる・やめる・整える” をゆるく占うツールです。")

    st.markdown("---")
    st.subheader("1. 声を録音 / 選択")
    st.write("※30秒〜1分程度の“ふつうに話している声”がおすすめです。起きてから1時間以上経ったあとの、普通の状態で話しかけてください。")

    # 🔹 ここだけ voice_age_checker.py と同じスタイルに
    wav_audio_data = st.audio_input("🎤 録音するか、音声ファイルを選択してください")

    if wav_audio_data is None:
        st.info("音声を録音（またはファイルを選択）すると分析が始まります。")
        return

    # 一時ファイルに保存
    audio_bytes = wav_audio_data.getbuffer()
    suffix = Path(wav_audio_data.name).suffix if wav_audio_data.name else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(tmp_path)

    with st.spinner("声を分析しています..."):
        try:
            features = extract_voice_features(tmp_path)
            scores = score_from_features(features)
            condition_text = describe_condition(scores)
            activities = judge_activity(scores)
            lucky = pick_lucky_item(scores)

            fashion = suggest_fashion(scores)
            lunch = suggest_lunch(scores)
            snack = suggest_snack(scores)
            day_plan = suggest_day_plan(scores)
        except Exception as e:
            st.error(f"分析中にエラーが発生しました: {e}")
            returndef main():
    st.set_page_config(page_title="声でわかるあなたの今日", page_icon="🎙", layout="centered")

    st.title("🎙 声でわかるあなたの今日")
    st.caption("声のざっくり特徴から、今日のコンディションと “やる・やめる・整える” をゆるく占うツールです。")

    st.markdown("---")
    st.subheader("1. 声を録音 / 選択")
    st.write("※30秒〜1分程度の“ふつうに話している声”がおすすめです。起きてから1時間以上経ったあとの、普通の状態で話しかけてください。")

    # 🔹 ここだけ voice_age_checker.py と同じスタイルに
    wav_audio_data = st.audio_input("🎤 録音するか、音声ファイルを選択してください")

    if wav_audio_data is None:
        st.info("音声を録音（またはファイルを選択）すると分析が始まります。")
        return

    # 一時ファイルに保存
    audio_bytes = wav_audio_data.getbuffer()
    suffix = Path(wav_audio_data.name).suffix if wav_audio_data.name else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(tmp_path)

    with st.spinner("声を分析しています..."):
        try:
            features = extract_voice_features(tmp_path)
            scores = score_from_features(features)
            condition_text = describe_condition(scores)
            activities = judge_activity(scores)
            lucky = pick_lucky_item(scores)

            fashion = suggest_fashion(scores)
            lunch = suggest_lunch(scores)
            snack = suggest_snack(scores)
            day_plan = suggest_day_plan(scores)
        except Exception as e:
            st.error(f"分析中にエラーが発生しました: {e}")
            return



        st.markdown("---")
        st.subheader("2. 今日の声コンディション")
        st.text(condition_text)

        # スコアの表示
        st.markdown("#### コンディションスコア（0〜1）")

        col1, col2, col3 = st.columns(3)
        col1.metric("元気さ（energy）", f"{scores['energy']:.2f}")
        col2.metric("緊張感（tension）", f"{scores['tension']:.2f}")
        col3.metric("安定感（stability）", f"{scores['stability']:.2f}")

        col4, col5, col6, col7 = st.columns(4)
        col4.metric("明るさ（brightness）", f"{scores['brightness']:.2f}")
        col5.metric("ザラつき（roughness）", f"{scores['roughness']:.2f}")
        col6.metric("抑揚（pitch_var）", f"{scores['pitch_var']:.2f}")
        col7.metric("テンポ（tempo）", f"{scores['tempo_index']:.2f}")

        st.markdown("---")
        st.subheader("3. 今日の“やる・やめる”診断")

        for name, info in activities.items():
            with st.expander(f"{name}：{info['label']}"):
                st.write(f"おすすめ度スコア: **{info['score']:.2f}**")
                st.write(info["advice"])

        st.markdown("---")
        st.subheader("4. 今日のラッキーアイテム")
        st.markdown(f"**ラッキーアイテム：{lucky['item']}**")
        st.caption(lucky["reason"])

        st.markdown("---")
        st.subheader("5. 今日のおすすめスタイル＆フード")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("### 👗 今日のファッション")
            st.markdown(f"**{fashion['title']}**")
            st.write(fashion["detail"])

        with col_f2:
            st.markdown("### 🍽 今日のお昼ごはん")
            st.markdown(f"**{lunch['title']}**")
            st.write(lunch["detail"])

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("### 🍪 今日のおやつ")
            st.markdown(f"**{snack['title']}**")
            st.write(snack["detail"])

        with col_s2:
            st.markdown("### 🕰 今日の過ごし方")
            st.markdown(f"**{day_plan['title']}**")
            st.write(day_plan["detail"])

        st.markdown("---")
        st.caption("※あくまで“ゆる診断”です。医療・採用などのシリアスな判断には使わないでください。")


if __name__ == "__main__":
    main()
