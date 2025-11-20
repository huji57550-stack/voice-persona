# voice_dinner_recommender.py
# 🍚 声でわかる 今日のあなたにおすすめの夕飯（主婦向け）
# 1. 声コンディション分析
# 2. 声フレーバー分析（メルヘン感・異国情緒・生活感など）
# 3. 1+2 を総合し、“今日のムード”を大げさに解釈して今夜のおかず一品を決定

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


# ========= 声の“フレーバー”分析（メルヘン感・異国情緒など） =========
def analyze_voice_flavors(traits: dict):
    """
    声の性質から、「世界観」を感じるフレーバーをいくつか付ける。
      例）メルヘン感 / 異国情緒 / 生活感 / 夜ふかし読書感 / 体育会系 / ヒーリング / ご褒美スイッチ
    戻り値：
      - flavors: [{title, desc}, ...]
      - flags  : dict（各フレーバーのON/OFFフラグ）
    """
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    flavors = []
    flags = {
        "is_fantasy": False,
        "is_exotic": False,
        "is_life": False,
        "is_nightowl": False,
        "is_sports": False,
        "is_healing": False,
        "is_reward": False,
    }

    # メルヘン感：明るめ＋抑揚大きめ＋テンポややゆっくり
    if b > 0.6 and pv > 0.55 and ti < 0.7:
        flags["is_fantasy"] = True
        flavors.append({
            "title": "メルヘン感がちょっと強めの声",
            "desc": (
                "明るさと抑揚がほどよくあって、しゃべり方にも少し“物語っぽさ”があります。\n"
                "絵本の読み聞かせや、子どもに今日あったことを話すシーンが似合う声です。"
            )
        })

    # 異国情緒：明るさ or ザラつきが高め＋テンポ速め or抑揚大きめ
    if (b > 0.65 or r > 0.6) and (ti > 0.55 or pv > 0.55):
        flags["is_exotic"] = True
        flavors.append({
            "title": "どこか異国情緒のある声",
            "desc": (
                "テンポや響きに少しクセがあって、“どこか別の国の空気”を感じさせる声です。\n"
                "旅先のカフェで隣の席から聞こえてきたら、思わず耳を傾けてしまいそうなタイプ。"
            )
        })

    # 生活感：エネルギー中くらい＋安定感そこそこ＋テンポも中くらい
    life_score = (abs(e - 0.5) * -1 + 1) * 0.4 + s * 0.3 + (1 - abs(ti - 0.5) * 2) * 0.3
    if life_score > 0.55:
        flags["is_life"] = True
        flavors.append({
            "title": "いい意味で“生活感”のある声",
            "desc": (
                "元気すぎず、静かすぎず、毎日の暮らしの中になじむ声です。\n"
                "家族との「ただいま」「おかえり」や、晩ごはんの「いただきます」が似合う、"
                "あたたかい生活音に近い存在。"
            )
        })

    # 夜ふかし読書感：テンポ遅め＋明るさ控えめ＋安定感高め
    if ti < 0.45 and b < 0.55 and s > 0.6:
        flags["is_nightowl"] = True
        flavors.append({
            "title": "夜ふかし読書が似合う声",
            "desc": (
                "テンポがゆっくりで、トーンも落ち着きめ。\n"
                "寝る前の読み聞かせや、夜の小さな独り言が絵になるタイプの声です。"
            )
        })

    # 体育会系：エネルギー高め＋テンポ速め＋ザラつき少し
    if e > 0.65 and ti > 0.6 and r > 0.35:
        flags["is_sports"] = True
        flavors.append({
            "title": "体育会系の“がんばって！”が似合う声",
            "desc": (
                "元気さと勢いがあって、少しだけザラっとした力強さも感じられる声です。\n"
                "運動会の応援や、「今日も一日がんばろ〜！」の一言がよく似合うタイプ。"
            )
        })

    # ふんわりヒーリング系：エネルギー低〜中・安定高・明るさ中くらい・抑揚控えめ
    if e < 0.6 and s > 0.65 and 0.3 < b < 0.7 and pv < 0.6:
        flags["is_healing"] = True
        flavors.append({
            "title": "ふんわりヒーリング系の声",
            "desc": (
                "大きな起伏はないけれど、ずっと聞いていられる安心感のある声です。\n"
                "疲れた日の「おつかれさま」が、じんわり心にしみこむタイプ。"
            )
        })

    # ご褒美スイッチ：明るさ高 or 抑揚高 ＋ テンポそこそこ
    if (b > 0.6 or pv > 0.6) and 0.4 < ti < 0.8:
        flags["is_reward"] = True
        flavors.append({
            "title": "“ご褒美スイッチ”を押してくれる声",
            "desc": (
                "どこかワクワク感があって、「今日くらいはいいよね」が似合う声です。\n"
                "アイスやデザート、外食を決めたときのテンションと相性抜群。"
            )
        })

    # 何も引っかからなかったときの保険
    if not flavors:
        flavors.append({
            "title": "バランスのとれたニュートラルな声",
            "desc": (
                "大きなクセが少なく、いろんな場面・いろんな相手に馴染みやすい声です。\n"
                "その日の気分や話す内容によって、いろんな雰囲気に化けられる“オールラウンダー”タイプ。"
            )
        })

    return {"flavors": flavors[:3], "flags": flags}


# ========= 料理そのものと声の性格の相性を説明する =========
def narrative_reason(traits, flags, menu_name):
    """
    料理の「性格」と声の「性格」を紐付けて、なぜその料理が合うのかを説明する。
    """
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]
    f = flags

    parts = []

    # --- 声の熱量と調理法の関係（大げさ解釈） ---
    if ti > 0.65:
        parts.append("今日は声のテンポが軽やかなので、火加減や焼き目に“リズム”が出る料理と相性が良さそうです。")
    elif ti < 0.45:
        parts.append("声のテンポが少しゆっくりめなので、時間をかけて味がしみ込む料理がしっくりきます。")

    if e > 0.7:
        parts.append("声にエネルギーがしっかりあるので、“ジュッ”と音がする香ばしい調理が今日のムードにハマります。")
    elif e < 0.4:
        parts.append("声のエネルギーが控えめな今日は、胃腸にも心にも優しい、負担の少ない調理法が合います。")

    # --- 声質と食材のキャラクターの相性 ---
    if b > 0.65:
        parts.append("声が明るめなので、旨味が前に出て香り立つような主役級食材が似合います。")
    elif b < 0.45:
        parts.append("トーンが落ち着いているので、じんわりしみるような味わいの料理とよく馴染みます。")

    if pv > 0.6:
        parts.append("抑揚がしっかりある声なので、味や食感に“表情”がある料理との相性が良い日です。")
    elif pv < 0.4:
        parts.append("抑揚少なめで穏やかな声なので、味にブレが少ない“安定感のあるおかず”が心地よく感じられます。")

    # --- フレーバーとの対応 ---
    if f["is_fantasy"]:
        parts.append("メルヘンなフレーバーがにじんでいるので、“開いた瞬間に楽しい”“香りで世界観が変わる”ような料理が似合います。")
    if f["is_exotic"]:
        parts.append("どこか異国情緒のある声なので、少しだけ旅っぽさを感じる味や香りとの相性が抜群です。")
    if f["is_life"]:
        parts.append("いい意味で生活感のある声なので、家族の食卓に自然に溶け込む“定番のおかず”がいちばんしっくりきます。")
    if f["is_healing"]:
        parts.append("ヒーリング系の声が出ている今日は、疲れた体をじんわり受けとめてくれる料理が合います。")
    if f["is_sports"]:
        parts.append("体育会系の勢いがあるので、“噛んで満足感のある肉料理”ととても相性のいいコンディションです。")
    if f["is_reward"]:
        parts.append("ご褒美スイッチが入っている声なので、一口でテンションが上がるおかずが今日のムードに合います。")

    # --- 料理別の補足（この料理である必然性） ---
    extra = ""
    if menu_name == "チキンソテー":
        extra = (
            "チキンソテーは、外はカリッと中はジューシーという“二面性”のある料理です。\n"
            "今日のあなたの声も、表はしっかり頑張っていて、内側は少し休みたい感じが混ざっているように聞こえるので、"
            "このギャップのある料理がよく似合います。"
        )
    elif menu_name == "鮭のホイル焼き":
        extra = (
            "鮭のホイル焼きは、包まれていたものを“ふわっ”と開いた瞬間に香りと湯気が広がる料理です。\n"
            "今日は声にも、心の中の本音や疲れをそっと包んでいるようなニュアンスがあるので、"
            "この“やさしく包んでから開く”料理がフィットします。"
        )
    elif menu_name == "鶏のから揚げ":
        extra = (
            "鶏のから揚げは、噛んだ瞬間に元気が湧いてくる“お祭り系”のおかずです。\n"
            "声に勢いや明るさがにじんでいる今日は、このストレートなパワーの塊みたいな料理が、"
            "あなたのテンションとぴったり重なります。"
        )
    elif menu_name == "さばの味噌煮":
        extra = (
            "さばの味噌煮は、コトコトと時間をかけて味をしみこませる、時間軸の長い料理です。\n"
            "今日は声にも安定感や“ゆっくり落ち着きたい”空気を感じるので、"
            "短距離走ではなくマラソンのようにじんわり効くこのおかずがよく合います。"
        )
    elif menu_name == "豚肉と野菜のオイスター炒め":
        extra = (
            "豚肉と野菜のオイスター炒めは、家庭料理の中でも少し“旅っぽい香り”がするポジションのメニューです。\n"
            "異国情緒やメルヘン感が今日の声から読み取れるので、"
            "いつものキッチンでちょっとだけ海外気分になれるこの料理がちょうどいい非日常になります。"
        )
    elif menu_name == "豚バラと白菜の鍋":
        extra = (
            "豚バラと白菜の鍋は、具材同士がゆっくり溶け合っていく“調和タイプ”の料理です。\n"
            "声に緊張や疲れの影が出ている日は、あれこれ味を変えるより、"
            "一つの鍋を囲んで落ち着けるこのメニューが、今のあなたに寄り添います。"
        )
    elif menu_name == "鶏の照り焼き":
        extra = (
            "鶏の照り焼きは、“甘じょっぱさ”とテリで安心感をくれる定番おかずです。\n"
            "生活感やバランスの良さが声から感じ取れる今日は、"
            "奇をてらわずに家族の“いつものおいしい”をちゃんと支えるこの料理がベストです。"
        )

    parts.append(extra)

    return "\n".join(parts)


# ========= 総合診断：その日のムードを大げさ解釈して夕飯を決める =========
def choose_dinner(traits: dict, flavor_flags: dict) -> dict:
    """
    声の状態（traits）＋ フレーバー（flavor_flags）を総合して、
    今日つくると良さそうなおかず一品を決定（主婦向け）。

    ※ 元気 or 疲れだけに寄らないよう、
       休みたい度・お祝いしたい度・こもって落ち着きたい度・旅気分度・生活感度など
       複数軸の“今日のムード”を作って、それをわざと大げさに解釈して決める。
    """
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    f = flavor_flags
    is_fantasy = f["is_fantasy"]
    is_exotic = f["is_exotic"]
    is_life = f["is_life"]
    is_nightowl = f["is_nightowl"]
    is_sports = f["is_sports"]
    is_healing = f["is_healing"]
    is_reward = f["is_reward"]

    # ==== 今日のムード指標（0〜1）を作る ====
    # “大げさに読む”ため、あえて強調ぎみに計算
    rest_need = float(np.clip((1 - e) * 0.5 + t * 0.3 + (1 - ti) * 0.2, 0, 1))
    celebrate_need = float(np.clip(e * 0.3 + b * 0.3 + pv * 0.2 + (0.2 if is_reward else 0), 0, 1))
    cozy_need = float(np.clip(s * 0.4 + (1 - b) * 0.2 + (0.2 if is_nightowl else 0) + (0.2 if is_healing else 0), 0, 1))
    travel_mood = float(np.clip((0.6 if is_exotic else 0) + (0.2 if is_fantasy else 0) + b * 0.2, 0, 1))
    life_mood = float(np.clip((0.4 if is_life else 0) + (1 - abs(ti - 0.5) * 2) * 0.3 + s * 0.3, 0, 1))

    # 軽い“日替わりニュアンス”を出すためのシード（同じ声でも少しだけ揺らぎやすくする）
    seed_base = e * 13 + t * 17 + s * 19 + b * 23 + pv * 29 + ti * 31
    seed = seed_base - int(seed_base)
    noise_scale = 0.04  # メニューが変わるか変わらないか、ギリギリくらいの小さな揺らぎ

    # ==== 各メニューのスコアを計算 ====
    menu_scores = {}

    # 1) 豚バラと白菜の鍋：休みたい + こもって落ち着きたい
    menu_scores["豚バラと白菜の鍋"] = (
        rest_need * 0.6
        + cozy_need * 0.3
        + life_mood * 0.1
        + (seed - 0.5) * noise_scale
    )

    # 2) 鮭のホイル焼き：少し省エネ＋メルヘン or 生活感
    menu_scores["鮭のホイル焼き"] = (
        rest_need * 0.4
        + cozy_need * 0.2
        + (0.2 if is_fantasy else 0)
        + life_mood * 0.2
        + (seed - 0.5) * (-noise_scale)  # 微妙に逆方向に揺らす
    )

    # 3) 鶏のから揚げ：お祝いしたい + 体育会系 / ご褒美
    menu_scores["鶏のから揚げ"] = (
        celebrate_need * 0.6
        + (0.2 if is_sports else 0)
        + (0.1 if is_reward else 0)
        + rest_need * 0.1
        + (seed - 0.5) * noise_scale
    )

    # 4) 豚肉と野菜のオイスター炒め：旅気分 + お祝いしたい
    menu_scores["豚肉と野菜のオイスター炒め"] = (
        travel_mood * 0.7
        + celebrate_need * 0.2
        + life_mood * 0.1
        + (seed - 0.5) * (-noise_scale)
    )

    # 5) さばの味噌煮：こもって落ち着きたい + 安定感 + 生活感
    menu_scores["さばの味噌煮"] = (
        cozy_need * 0.5
        + life_mood * 0.3
        + rest_need * 0.2
        + (seed - 0.5) * noise_scale
    )

    # 6) チキンソテー：シンプルに焼いて、それなりにちゃんとしたい日
    simple_axis = float(np.clip(s * 0.4 + (1 - pv) * 0.3 + (1 - t) * 0.3, 0, 1))
    menu_scores["チキンソテー"] = (
        simple_axis * 0.5
        + rest_need * 0.2
        + celebrate_need * 0.1
        + travel_mood * 0.2
        + (seed - 0.5) * (-noise_scale)
    )

    # 7) 鶏の照り焼き：生活感 + バランス型
    menu_scores["鶏の照り焼き"] = (
        life_mood * 0.6
        + cozy_need * 0.2
        + celebrate_need * 0.2
        + (seed - 0.5) * noise_scale
    )

    # スコア最大のメニューを選ぶ
    menu_name = max(menu_scores.items(), key=lambda x: x[1])[0]

    # ==== メニューごとの文章（summary / tips / selfcare） ====
    if menu_name == "豚バラと白菜の鍋":
        summary = "今日は“がんばりすぎた日寄り”の声。コンロに張りつかなくて済む、やさしい鍋にしましょう。"
        reason = (
            "休みたい度と、こもって落ち着きたい度が高めに出ています。"
            "一方で、生活感の軸もそこそこ強いので、“ちゃんとご飯は用意したい”気持ちも感じられます。\n"
            "豚バラと白菜の鍋は、切って重ねて煮るだけで、手間は低いのに食卓としての満足度は高い、"
            "今日のムードにぴったりのバランス型鍋です。"
        )
        tips = (
            "白菜と豚バラを交互に重ねて鍋に入れ、だし・酒・しょうゆ少々でコトコト煮るだけ。\n"
            "ポン酢やごまだれ、ラー油などをテーブルに並べて、各自が好きな味で食べられるようにすると、"
            "“鍋一品”でも満足感が上がります。"
        )
        selfcare = (
            "今日は“料理で点数を取りに行く日”ではなく、“ちゃんと休んで明日に備える日”です。\n"
            "鍋が煮えているあいだ、イスに座って背もたれに体重を預ける時間を、自分に許してください。"
        )

    elif menu_name == "鮭のホイル焼き":
        summary = "今日は“静かにやり過ごしたい日寄り”の声。包んで焼くだけのホイル焼きで、体力温存モードに。"
        reason = (
            "休みたい度はそこそこ高く、こもって落ち着きたい度も強め。"
            "同時に、生活感フレーバーもあるので、“家っぽい安心感”のある料理がしっくりきます。\n"
            "鮭のホイル焼きは、包んでしまえばあとは火にかけるだけ、"
            "それでも“ちゃんと料理した感”が出る、今日のあなたにやさしいメニューです。"
        )
        tips = (
            "アルミホイルに鮭と野菜（玉ねぎ・きのこ・人参など）をのせて包んで焼くだけ。\n"
            "味付けは、塩こしょう＋バター、または味噌＋みりんを少しのせるだけでもOKです。\n"
            "1人分ずつ包めば、そのままお皿にのせて洗い物も最小限にできます。"
        )
        selfcare = (
            "ホイルを開けたときの湯気と香りは、そのまま“今日一日よくがんばった自分へのご褒美”です。\n"
            "「これで十分」と、心の中でそっと自分に言ってあげてください。"
        )

    elif menu_name == "鶏のから揚げ":
        summary = "声に元気とご褒美感がにじんでいる日。今日は“正解のない日々”を一旦置いて、唐揚げでお祭りモードに。"
        reason = (
            "お祝いしたい度が高く、体育会系・ご褒美フレーバーもにじんでいる日です。\n"
            "こういう日は、栄養バランスよりも、“みんなのテンションを一気に上げる一皿”を置くのが正解です。\n"
            "鶏のから揚げは、まさにその役。食卓の空気を一瞬で“やった！”に変えてくれます。"
        )
        tips = (
            "鶏もも肉を一口大に切り、しょうゆ・酒・にんにく・生姜で下味をつけて片栗粉をまぶし、油で揚げます。\n"
            "揚げ物がしんどい日は、フライパンで少なめの油を使って“揚げ焼き”でもOK。\n"
            "キャベツの千切りを敷いて、レモンやマヨネーズを添えれば、見た目も満足度も一気に上がります。"
        )
        selfcare = (
            "から揚げの日は、家族や自分の「おいしい！」をしっかり受け取ってください。\n"
            "それは今日のあなたが、“ちゃんと生活を回した証拠”でもあります。"
        )

    elif menu_name == "豚肉と野菜のオイスター炒め":
        summary = "声にちょっと旅心とワクワクが混ざっている日。今日はキッチンでささやかな“異国感”を味わいましょう。"
        reason = (
            "旅気分・非日常度が高めに出ていて、同時に“少しテンション上げたい”空気も感じられます。\n"
            "でも、外食に行くほどのエネルギーは使いたくない…そんな日の落としどころが、"
            "フライパンひとつで作れるオイスター炒めです。"
        )
        tips = (
            "豚こま肉と野菜（キャベツ・ピーマン・もやし・玉ねぎなど）を炒め、\n"
            "オイスターソース・しょうゆ・酒・砂糖少々で味付けするだけ。\n"
            "ご飯にのせて丼にしたり、翌日のお弁当おかずにも回せる、影の優等生メニューです。"
        )
        selfcare = (
            "“どこかに行きたいけど、今日は家にいたい”という気分も、立派なコンディションの一つです。\n"
            "キッチンで少しだけ異国感を足してあげることで、心の「どこかに行きたい」をなだめてあげましょう。"
        )

    elif menu_name == "さばの味噌煮":
        summary = "声に落ち着きとヒーリング感がにじむ日。今日は“いつもの和食”で、静かに体と心を整える夜に。"
        reason = (
            "こもって落ち着きたい度が高く、安定感やヒーリング系のフレーバーも強めです。\n"
            "華やかな料理よりも、味が決まっていて、食べたときにホッとするような定番和食が合う日です。\n"
            "さばの味噌煮は、時間とともにおいしさが増していく“じわじわ効くおかず”なので、今日の声とよく噛み合います。"
        )
        tips = (
            "さばに熱湯をかけて臭みをとり、味噌・砂糖・酒・みりん・しょうゆ・生姜でコトコト煮るだけ。\n"
            "時間さえかければ、多少味付けがラフでもそれなりにおいしく仕上がるのも魅力です。\n"
            "ねぎや大根を一緒に煮ると、さらに満足度がアップします。"
        )
        selfcare = (
            "“特別感”はなくても、「今日もちゃんと家でご飯を食べた」という事実が、"
            "じわじわあなたの体を守っています。\n"
            "食卓に座った瞬間に、一度肩の力を抜いて、深呼吸をしてみてください。"
        )

    elif menu_name == "チキンソテー":
        summary = "今日は声も気分も“ほどよく頑張れている日”。焼くだけで映えるチキンソテーで、手間と満足度のバランスを取る夜に。"
        reason = (
            "シンプルさ・安定感の軸が高めで、“すごくお祝いしたいわけでも、完全に休みモードでもない”空気です。\n"
            "そんな日は、凝った料理よりも、“焼くだけでそれっぽく見える”メインが最適解。\n"
            "チキンソテーは、まさにそのポジションで、今日の声のニュートラルさとぴったり合います。"
        )
        tips = (
            "鶏もも肉に塩こしょうをして、皮目からじっくり焼くだけ。\n"
            "ふたをして中まで火を通し、最後にしょうゆを少し垂らすと、ご飯に合う味になります。\n"
            "付け合わせは、冷凍フライドポテトや温野菜を並べるだけでも“ちゃんとして見える”のでOK。"
        )
        selfcare = (
            "“とりあえず今日もここまでやった”という感覚を大事にしてほしい日です。\n"
            "チキンソテーが焼き上がったら、それを合図に、今日の自分にいったん終業ボタンを押してあげてください。"
        )

    else:  # 鶏の照り焼き
        menu_name = "鶏の照り焼き"
        summary = "今日は声も気分も“日常モード”。明日のお弁当にも回せる、安心の鶏の照り焼きが一番しっくり。"
        reason = (
            "生活感・バランス型のムードが強く、“派手さよりも安定感”が欲しい日です。\n"
            "こういう日は、家族にとっても自分にとっても“ハズレなし”の定番おかずが、"
            "いちばん心を落ち着かせてくれます。\n"
            "鶏の照り焼きは、まさにその王道ポジションです。"
        )
        tips = (
            "鶏もも肉を焼いて、しょうゆ・みりん・砂糖・酒を絡めるだけの王道レシピ。\n"
            "タレを少し多めにして、翌日の弁当おかずに使い回すと、未来の自分も助かります。"
        )
        selfcare = (
            "“いつもの照り焼き”を出せる日は、実はかなりコンディションが良い証拠です。\n"
            "「特別なことはしてないけど、家は回している」その事実を、ちゃんと評価してあげてください。"
        )

    # 料理そのものと声の性格の相性を、別軸でさらに説明
    narrative = narrative_reason(traits, flavor_flags, menu_name)

    return {
        "name": menu_name,
        "summary": summary,
        "reason": reason + "\n\n---\n### 🔍 料理そのものとの相性の理由\n" + narrative,
        "tips": tips,
        "selfcare": selfcare,
    }


# ========= 声コンディションのテキスト =========
def explain_traits(traits: dict) -> str:
    e = traits["energy"]
    t = traits["tension"]
    s = traits["stability"]
    b = traits["brightness"]
    r = traits["roughness"]
    pv = traits["pitch_var"]
    ti = traits["tempo_index"]

    lines = []
    lines.append("● 今日の声から見たコンディション（0〜1のざっくり指標）")
    lines.append(f"- 元気さ（energy）      : {e:.2f}")
    lines.append(f"- 緊張感（tension）     : {t:.2f}")
    lines.append(f"- 安定感（stability）   : {s:.2f}")
    lines.append(f"- 明るさ（brightness）  : {b:.2f}")
    lines.append(f"- ザラつき（roughness） : {r:.2f}")
    lines.append(f"- 抑揚（pitch_var）      : {pv:.2f}")
    lines.append(f"- テンポ（tempo_index） : {ti:.2f}")
    lines.append("")
    lines.append("※これはあくまで音声特徴を使ったエンタメ診断です。")
    lines.append("　“今日の気分をどう扱うか”を考えるヒントとして、気軽に楽しんでください。")
    return "\n".join(lines)


# ========= Streamlit UI =========
def main():
    st.set_page_config(
        page_title="声でわかる 今日のあなたにおすすめの夕飯",
        page_icon="🍚",
        layout="centered",
    )

    st.title("🍚 声でわかる 今日のあなたにおすすめの夕飯")
    st.caption(
        "毎日使う前提で、“その日の声”からムードを少し大げさに読み取り、"
        "今夜のおかず一品を決める主婦向けエンタメ診断です。"
    )

    st.markdown("---")
    st.subheader("1. 声の録音")

    st.write(
        "普段どおりの話し方で、30秒〜1分ほど話した音声がおすすめです。\n"
        "今日あったことや、夕飯どうしようかな〜という本音をしゃべって録音してみてください。"
    )

    uploaded = st.audio_input("🎤 マイクで話して録音してください（30秒〜1分）")

    if uploaded is None:
        st.info("マイクで録音すると、診断ボタンが表示されます。")
        return

    # 一時ファイルに保存
    audio_bytes = uploaded.getbuffer()
    suffix = ".wav"   # audio_input は WAV 固定
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.markdown("#### 音声プレビュー")
    st.audio(tmp_path)

    if st.button("今日の分析と夕飯おかずを診断する"):
        with st.spinner("今日の声を分析して、今夜のメニューを考えています..."):
            try:
                features = extract_voice_features(tmp_path)
                traits = score_voice_traits(features)
                trait_text = explain_traits(traits)
                flavor_info = analyze_voice_flavors(traits)
                flavors = flavor_info["flavors"]
                flags = flavor_info["flags"]
                dinner = choose_dinner(traits, flags)
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                return

        # 2. コンディション
        st.markdown("---")
        st.subheader("2. 今日の声コンディション")
        st.text(trait_text)

        # 3. フレーバー診断
        st.markdown("---")
        st.subheader("3. 声の“雰囲気フレーバー”診断")
        for flavor in flavors:
            st.markdown(f"### 🌈 {flavor['title']}")
            st.write(flavor["desc"])
            st.markdown("")

        # 4. 総合診断：今夜のおかず
        st.markdown("---")
        st.subheader("4. 総合診断：今夜のおすすめおかず一品")

        # まずメニュー名だけを出す
        st.markdown(f"### 🥢 今夜のメニュー：**{dinner['name']}**")

        # 総合コメント
        st.markdown("#### 今日のあなたにこれが合う理由（総合コメント）")
        st.write(dinner["summary"])

        st.markdown("#### 声の分析から見た詳しい理由")
        st.write(dinner["reason"])

        st.markdown("#### 作るときのポイント")
        st.write(dinner["tips"])

        st.markdown("#### 今日のあなたへのひと言ケア")
        st.write(dinner["selfcare"])

        st.markdown("---")
        st.caption(
            "※この診断はエンタメ用です。栄養バランスや持病などがある場合は、"
            "　医師や管理栄養士など専門家のアドバイスもあわせて大事にしてくださいね。"
        )


if __name__ == "__main__":
    main()
