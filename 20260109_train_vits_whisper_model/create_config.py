import yaml
import os
from collections.abc import MutableMapping

# デフォルト設定ファイルと新しい設定ファイルのパスを定義
DEFAULT_CONFIG_PATH = "configs/default_config.yaml"
NEW_CONFIG_PATH = "configs/config.yaml"


def get_user_input_for_config(config_dict):
    """
    設定辞書を再帰的に処理し、ユーザーに新しい値を問い合わせ、
    更新された値を持つ新しい辞書を返します。
    """
    new_config = {}
    for key, value in config_dict.items():
        # 値がNone（YAMLで`~`）の場合は、有効にするかユーザーに尋ねる
        if value is None:
            if input(f"'{key}' を有効にして設定しますか？ (y/n, デフォルト: n): ").lower() != 'y':
                continue
            else:
                # 有効にする場合、ユーザーに値を尋ねる
                user_input = input(f"'{key}' の値を入力してください: ")
                # ユーザーが入力した値を、空文字列であっても設定する
                new_config[key] = user_input
                continue

        if isinstance(value, MutableMapping):
            # 値が辞書の場合は、再帰的に処理
            print(f"\n--- '{key}' セクションの設定 ---")
            new_config[key] = get_user_input_for_config(value)
        else:
            # ユーザーに新しい値を尋ねる
            prompt = f"'{key}' の値を入力してください (デフォルト: {value}): "
            user_input = input(prompt)
            # ユーザーが値を入力した場合はその値を、そうでなければデフォルト値を使用
            if user_input:
                # デフォルト値と同じ型に変換を試みる
                try:
                    # 真偽値の場合は特別に処理
                    if isinstance(value, bool):
                        if user_input.lower() in ['true', 't', 'yes', 'y', 'はい']:
                            new_config[key] = True
                        elif user_input.lower() in ['false', 'f', 'no', 'n', 'いいえ']:
                            new_config[key] = False
                        else:
                            # 曖昧な入力の場合はデフォルト値を維持
                            new_config[key] = value
                    else:
                        new_config[key] = type(value)(user_input)
                except (ValueError, TypeError):
                    # 型変換に失敗した場合は、文字列として保存
                    new_config[key] = user_input
            else:
                new_config[key] = value
    return new_config


def main():
    """
    対話形式で設定ファイルを生成するメイン関数。
    """
    print("--- 対話形式 設定セットアップ ---")
    print(f"デフォルト設定を {DEFAULT_CONFIG_PATH} から読み込んでいます...\n")

    # デフォルト設定ファイルが存在するか確認
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        print(f"エラー: デフォルト設定ファイルが {DEFAULT_CONFIG_PATH} に見つかりません。")
        return

    # デフォルト設定を読み込む
    with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        default_config = yaml.safe_load(f)

    # ユーザーから設定値を取得
    print("新しい値を入力するか、Enterキーを押してデフォルト値を使用してください。")
    new_config = get_user_input_for_config(default_config)

    # 保存先ディレクトリが存在することを確認
    print(f"設定ファイルを新たに設定しますか？ 保存先: {NEW_CONFIG_PATH}")
    if input("続行しますか？ (y/n): ").lower() != 'y':
        CONFIG_PATH = input("新しい保存先パスを入力してください: ")
    else:
        CONFIG_PATH = NEW_CONFIG_PATH
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    # 新しい設定を config.yaml に書き込む
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n設定ファイルが {CONFIG_PATH} に正常に保存されました。")


if __name__ == "__main__":
    main()
