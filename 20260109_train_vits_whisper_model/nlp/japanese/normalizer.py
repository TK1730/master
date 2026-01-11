import re
import unicodedata
from num2words import num2words
from nlp.symbols import PUNCTUATIONS


__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    # NFKC 正規化後のハイフン・ダッシュの変種を全て通常半角ハイフン - \u002d に変換
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen,
    # "\u2011": "\u002d",  # ‑, Non-Breaking Hyphen, NFKC により \u2010 に変換される
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # "~": "-",  # これも長音記号「ー」として扱うよう変更
    "「": "'",
    "」": "'",
}
__REPLACE_PATTERN = re.compile("|".join(re.escape(p) for p in __REPLACE_MAP))
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    # ひらがな、カタカナ、漢字
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    + r"\u0041-\u005A\u0061-\u007A"  # 半角アルファベット
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"  # 全角アルファベット (NFKCで半角になる想定)
    + r"\u0370-\u03FF\u1F00-\u1FFF"  # ギリシャ文字
    + "".join(re.escape(p) for p in PUNCTUATIONS)
    + r"]+",  # 許可する句読点（エスケープ済み）
)
__CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
__CURRENCY_PATTERN = re.compile(r"([$¥£€])([0-9.]*[0-9])")
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")

# より多くのハイフン/マイナス記号を半角ハイフンに正規化
__HYPHEN_PATTERN = re.compile(r"[˗֊‐‑‒–⁃⁻₋−]+")
# より多くの長音記号をカタカナ長音符に正規化
__CHOONPU_PATTERN = re.compile(r"[﹣－ｰ—―─━ー]+")
# チルダ類 (音声合成では長音として扱いたいことが多い)
__TILDE_PATTERN = re.compile(r"[~∼∾〜〰～]+")


def __convert_numbers_to_words(text: str) -> str:
    """
    記号や数字を日本語の文字表現に変換する

    Args:
        text (str): 変換対象のテキスト

    Returns:
        str: 変換後のテキスト
    """
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), text)
    res = __CURRENCY_PATTERN.sub(lambda m: m[2] + __CURRENCY_MAP.get(m[1], m[1]), res)
    res = __NUMBER_PATTERN.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def normalize_text(text: str) -> str:
    """
    日本語のテキストを正規化する
    結果
    - ひらがな
    - カタカナ（全角長音記号「ー」が入る！）
    - 漢字
    - 半角アルファベット（大文字と小文字）
    - ギリシャ文字
    - `.` （句点`。`や`…`の一部や改行等）
    - `,` （読点`、`や`:`等）
    - `?` （疑問符`？`）
    - `!` （感嘆符`！`）
    - `'` （`「`や`」`等）
    - `-` （`―`（ダッシュ、長音記号ではない）や`-`等）

    注意点:
    - 三点リーダー`…`は`...`に変換される（`なるほど…。` → `なるほど....`）
    - 数字は漢字に変換される（`1,100円` → `千百円`、`52.34` → `五十二点三四`）
    - 読点や疑問符等の位置・個数等は保持される（`??あ、、！！！` → `??あ,,!!!`）
    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    # Unicode 正規化
    res = unicodedata.normalize("NFKC", text)  # ここでアルファベットは半角になる
    # ハイフン正規化
    res = __HYPHEN_PATTERN.sub("-", res)
    # 長音記号とチルダの正規化
    res = __CHOONPU_PATTERN.sub("ー", res)
    res = __TILDE_PATTERN.sub("ー", res)  # チルダも長音記号として扱う
    # 数字を日本語の文字表現に変換
    res = __convert_numbers_to_words(res)  # 「100円」→「百円」等
    # 句読点等正規化、読めない文字を削除
    res = replace_punctuation(res)

    # 結合文字の濁点・半濁点を削除
    # 通常の「ば」等はそのままのこされる、「あ゛」は上で「あ゙」になりここで「あ」になる
    res = res.replace("\u3099", "")  # 結合文字の濁点を削除、る゙ → る
    res = res.replace("\u309a", "")  # 結合文字の半濁点を削除、な゚ → な
    return res


def replace_punctuation(text: str) -> str:
    """
    句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalk で読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、アルファベット、ギリシャ文字

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 句読点を辞書で置換
    replaced_text = __REPLACE_PATTERN.sub(lambda x: __REPLACE_MAP[x.group()], text)

    # 上述以外の文字を削除
    replaced_text = __PUNCTUATION_CLEANUP_PATTERN.sub("", replaced_text)

    return replaced_text


if __name__ == "__main__":
    test_text = (
        "これはﾃｽﾄです。No.１、2,345円〜　（株）Style‐Bert‐VITS2 V2.0 〽 AÏ あ゙あ゚？"
    )
    normalized = normalize_text(test_text)
    print(f"Original: {test_text}")
    print(f"Normalized: {normalized}")
    # 期待される出力例: Normalized: これはテストです'ナンバー一'二千三百四十五円ー
    # 'かぶしきがいしゃスタイル-バート-ヴィッツツー'ヴィーツーてんれい'エーアイ'ああ?'
