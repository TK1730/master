"""
transcription.csvをtrainとtestデータセットに分割するスクリプト

各データセットカテゴリから10%をテストセットとしてランダムに抽出し、
残りの90%をトレーニングセットとして保存します。
"""

import pandas as pd
import pathlib
import random


def main():
    """メイン関数：データセットを訓練用とテスト用に分割する"""
    # 再現性のために乱数シードを固定
    random.seed(42)

    # パスの設定
    base_dir = pathlib.Path.cwd()
    dataset_dir = base_dir / "dataset"
    transcription_file = dataset_dir / "transcription.csv"
    train_file = dataset_dir / "train.csv"
    test_file = dataset_dir / "test.csv"

    # 分割対象のカテゴリ
    # すべてのデータセットを対象にする場合はすべてコメントを外す
    categories = [
        "nonpara30",
        "nonpara30w_ver2",
        "whisper10",
        "throat_microphone"
    ]

    print(f"{transcription_file}を読み込み中...")
    try:
        df = pd.read_csv(transcription_file)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        return

    # カラムの確認
    if "audio_path" not in df.columns:
        print("エラー: 'audio_path'カラムが見つかりません")
        return

    print(f"総データ数: {len(df)}")

    test_indices = []
    all_category_indices = []

    # 各カテゴリから10%をランダムにサンプリング
    for category in categories:
        # audio_pathにカテゴリ名が含まれる行を抽出
        # パスの形式の違いに対応するため、バックスラッシュをスラッシュに正規化
        mask = df["audio_path"].apply(
            lambda x: category in str(x).replace("\\", "/")
        )
        cat_indices = df[mask].index.tolist()
        count = len(cat_indices)

        if count == 0:
            print(f"警告: カテゴリ'{category}'のサンプルが見つかりません")
            continue

        all_category_indices.extend(cat_indices)

        # 10%をテストセット用にサンプリング
        sample_size = int(count * 0.1)
        if sample_size == 0 and count > 0:
            sample_size = 1  # 最低1サンプルは確保

        selected_indices = random.sample(cat_indices, sample_size)
        test_indices.extend(selected_indices)

        print(f"カテゴリ'{category}': 総数 {count}件, "
              f"テストセット {sample_size}件選択")

    # 訓練セットとテストセットを作成
    # テストセット: 選択されたインデックス
    # 訓練セット: 指定されたカテゴリに属する他のすべてのインデックス

    # 重複を削除（念のため）
    test_indices = list(set(test_indices))

    # 対象カテゴリのみをフィルタリング
    filtered_df = df.loc[all_category_indices].drop_duplicates()

    # 分割を作成
    test_df = df.loc[test_indices]
    # 訓練データはフィルタリングしたデータからテストインデックスを除外
    train_df = filtered_df.drop(test_indices, errors='ignore')

    print("\n分割結果を保存中...")
    print(f"訓練セット: {len(train_df)}件")
    print(f"テストセット: {len(test_df)}件")
    print(f"合計: {len(train_df) + len(test_df)}件")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\n訓練データ: {train_file}")
    print(f"テストデータ: {test_file}")
    print("完了しました！")


if __name__ == "__main__":
    main()
