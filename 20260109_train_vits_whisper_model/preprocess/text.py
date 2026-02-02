import argparse
from pathlib import Path
import json
from collections import defaultdict
from random import sample
from typing import Optional

from tqdm import tqdm

from utils.config import get_config
from utils.logger import logger
from nlp import clean_text
from nlp.japanese import pyopenjtalk_worker
from nlp.japanese.user_dict import update_dict
from utils.stdout_wrapper import SAFE_STDOUT

# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data / 以下の辞書データを pyopenjtalk_worker に登録
update_dict()

preprocess_text_config = get_config().preprocess_text_config


# Count lines for tqdm
def count_lines(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_error_log(error_log_path: Path, line: str, error: Exception):
    with error_log_path.open("a", encoding="utf-8") as error_log:
        error_log.write(f"{line.strip()}\n{error}\n\n")


def process_line(
    line: str,
    transcription_path: Path,
    correct_path: bool,
    use_jp_extra: bool,
    yomi_error: str,
):
    splitted_line = line.strip().split("|")
    if len(splitted_line) != 4:
        raise ValueError(f"Invalid line format: {line.strip()}")
    utt, spk, language, text = splitted_line
    norm_text, phones, tones, word2ph = clean_text(
        text=text,
        language=language,  # type: ignore
        use_jp_extra=use_jp_extra,
        raise_yomi_error=(yomi_error != "use"),
    )
    if correct_path:
        utt = str(transcription_path.parent / "wavs" / utt)

    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt,
        spk,
        language,
        norm_text,
        " ".join(phones),
        " ".join([str(i) for i in tones]),
        " ".join([str(i) for i in word2ph]),
    )


def process_line_jvs(
    line: str,
    transcription_path: Path,
    correct_path: bool,
    use_jp_extra: bool,
    yomi_error: str,
) -> str:
    """transcription ファイルの1行を処理します。

    期待する入力形式:
        audio.wav|text

    `nlp.clean_text` でテキストを正規化し、音素・アクセント情報に変換した
    後、phones/tones/word->phoneme マッピングを付与したパイプ区切り文字列
    を返します。

    Args:
        line (str): transcription ファイルの1行。
        transcription_path (Path): transcription ファイルのパス。
            `correct_path` が True の場合、wavs への絶対パス作成に使用します。
        correct_path (bool): True のとき、line 中の音声ファイル名を相対パスと
            みなし `<transcription_parent>/wavs/` に変換します。
        use_jp_extra (bool): 日本語向けの追加ルールを適用するかどうか。
        yomi_error (str): 読めない文字の扱い（`clean_text` に渡します）。
            "use" の場合は例外を上げません。

    Raises:
        ValueError: line が4つのフィールド
            (audio|speaker|language|text) でない場合。

    Returns:
        str: 改行で終わる1行の文字列:
            audio|speaker|language|norm_text|phones|tones|word2ph

    使用例:
        >>> from pathlib import Path
        >>> line = "001.wav|spk01|今日はいい天気"
        >>> out = process_line(line, Path("data/trans.txt"), True, True, "use")
        >>> print(out)
        001.wav|spk01|ja|今日はいい天気|k o N n i ch i w a|0 0 0|['今日は','今日']\n
    注意:
        phones, tones, word2ph はスペースで結合されます。空リストは空
        フィールドとして出力されます。
    """
    splitted_line = line.strip().split(":")
    if len(splitted_line) != 2:
        raise ValueError(f"Invalid line format: {line.strip()}")

    utt, text = splitted_line
    language = "JP"
    spk = transcription_path.parent.parent.name
    print(spk)
    norm_text, phones, tones, word2ph = clean_text(
        text=text,
        language=language,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=(yomi_error != "use"),
    )

    if correct_path:
        utt = str(transcription_path.parent / "wavs" / utt)

    # Convert lists to readable strings; handle empty values safely
    phones_str = " ".join(phones) if phones else ""
    tones_str = " ".join(str(t) for t in tones) if tones else ""
    word2ph_str = " ".join(map(str, word2ph)) if word2ph else ""

    return (
        f"{utt}|{spk}|{language}|{norm_text}|{phones_str}|"
        f"{tones_str}|{word2ph_str}\n"
    )


def preprocess(
    transcription_path: Path,
    cleaned_path: Optional[Path],
    train_path: Path,
    val_path: Path,
    config_path: Path,
    val_per_lang: int,
    max_val_total: int,
    # clean: bool,
    use_jp_extra: bool,
    yomi_error: str,
    correct_path: bool,
):
    assert yomi_error in ["raise", "skip", "use"]
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path.with_name(
            transcription_path.name + ".cleaned"
        )

    error_log_path = transcription_path.parent / "text_error.log"

    if error_log_path.exists():
        error_log_path.unlink()
    error_count = 0

    total_lines = count_lines(transcription_path)

    # transcription_path から 1行ずつ読み込んで文章処理して cleaned_path に書き込む
    with (
        transcription_path.open("r", encoding="utf-8") as trans_file,
        cleaned_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in tqdm(
            trans_file, file=SAFE_STDOUT, total=total_lines, dynamic_ncols=True
        ):
            try:
                processed_line = process_line(
                    line,
                    transcription_path,
                    correct_path,
                    use_jp_extra,
                    yomi_error,
                )
                out_file.write(processed_line)
            except Exception as e:
                logger.error(
                    f"An error occurred at line:\n{line.strip()}\n{e}", encoding="utf-8"
                )
                write_error_log(error_log_path, line, e)
                error_count += 1

    transcription_path = cleaned_path

    # 各話者ごとのlineの辞書
    spk_utt_map: dict[str, list[str]] = defaultdict(list)

    # 話者からIDへの写像
    spk_id_map: dict[str, int] = {}

    # 話者ID
    current_sid: int = 0

    # 音源ファイルのチェックや、spk_id_mapの作成
    with transcription_path.open("r", encoding="utf-8") as f:
        audio_paths: set[str] = set()
        count_same = 0
        count_not_found = 0
        for line in f.readlines():
            utt, spk = line.strip().split("|")[:2]
            if utt in audio_paths:
                logger.warning(f"Same audio file appears multiple times: {utt}")
                count_same += 1
                continue
            if not Path(utt).is_file():
                logger.warning(f"Audio not found: {utt}")
                count_not_found += 1
                continue
            audio_paths.add(utt)
            spk_utt_map[spk].append(line)

            # 新しい話者が出てきたら話者IDを割り当て、current_sidを1増やす
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1
        if count_same > 0 or count_not_found > 0:
            logger.warning(
                f"Total repeated audios: {count_same}, "
                f"Total number of audio not found: {count_not_found}"
            )

    train_list: list[str] = []
    val_list: list[str] = []

    # 各話者ごとに発話リストを処理
    for spk, utts in spk_utt_map.items():
        if val_per_lang == 0:
            train_list.extend(utts)
            continue
        # ランダムにval_per_lang個のインデックスを選択
        n_val = min(len(utts), val_per_lang)
        val_indices = set(sample(range(len(utts)), n_val))
        # 元の順序を保ちながらリストを分割
        for index, utt in enumerate(utts):
            if index in val_indices:
                val_list.append(utt)
            else:
                train_list.append(utt)

    # バリデーションリストのサイズ調整
    if len(val_list) > max_val_total:
        extra_val = val_list[max_val_total:]
        val_list = val_list[:max_val_total]
        # 余剰のバリデーション発話をトレーニングリストに追加（元の順序を保持）
        train_list.extend(extra_val)

    # Append to existing train/val files instead of overwriting them
    with train_path.open("a", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with val_path.open("a", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    with config_path.open("r", encoding="utf-8") as f:
        json_config = json.load(f)

    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    if error_count > 0:
        if yomi_error == "skip":
            logger.warning(
                f"An error occurred in {error_count} lines. Proceed with lines without errors. Please check {error_log_path} for details."
            )
        else:
            # yom_error == "raise"と"use"の場合。
            # "use"の場合は、そもそもyomi_error = Falseで処理しているので、
            # ここが実行されるのは他の例外のときなので、エラーをraiseする。
            logger.error(
                f"An error occurred in {error_count} lines. Please check {error_log_path} for details."
            )
            raise Exception(
                f"An error occurred in {error_count} lines. Please check `Data/you_model_name/text_error.log` file for details."
            )
            # 何故か{error_log_path}をraiseすると文字コードエラーが起きるので上のように書いている
    else:
        logger.info(
            "Training set and validation set generation from texts is complete!"
        )


if __name__ == "__main__":
    from preprocess import transcription
    from utils.config import TextPrompter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcription-path", default=preprocess_text_config.transcription_path
    )
    parser.add_argument("--cleaned-path", default=preprocess_text_config.cleaned_path)
    parser.add_argument("--train-path", default=preprocess_text_config.train_path)
    parser.add_argument("--val-path", default=preprocess_text_config.val_path)
    parser.add_argument("--config-path", default=preprocess_text_config.config_path)

    # 「話者ごと」のバリデーションデータ数、言語ごとではない！
    # 元のコードや設定ファイルでval_per_langとなっていたので名前をそのままにしている
    parser.add_argument(
        "--val-per-lang",
        default=preprocess_text_config.val_per_lang,
        help=(
            "Number of validation data per SPEAKER, not"
            " per language (due to compatibility with the original code)."
        ),
    )
    parser.add_argument("--max-val-total", default=preprocess_text_config.max_val_total)
    parser.add_argument("--use_jp_extra", action="store_true")
    parser.add_argument("--yomi_error", default="raise")
    parser.add_argument("--correct_path", action="store_true")

    args = parser.parse_args()
    prompter = TextPrompter()
    args = prompter.prompt(args)

    transcript = Path(args.transcription_path)
    cleaned_path = Path(args.cleaned_path) if args.cleaned_path else None
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    config_path = Path(args.config_path)
    val_per_lang = int(args.val_per_lang)
    max_val_total = int(args.max_val_total)
    use_jp_extra: bool = args.use_jp_extra
    yomi_error: str = args.yomi_error
    correct_path: bool = args.correct_path

    output_dir = Path("preprocessed/VITS_pretrain")
    dataset_path = Path("preprocessed/VITS_pretrain")

    # jvsデータセットの各話者ディレクトリを走査
    preprocess(
        transcription_path=transcript,
        cleaned_path=cleaned_path,
        train_path=train_path,
        val_path=val_path,
        config_path=config_path,
        val_per_lang=val_per_lang,
        max_val_total=max_val_total,
        use_jp_extra=use_jp_extra,
        yomi_error=yomi_error,
        correct_path=False,
    )
