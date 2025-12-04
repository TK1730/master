# **音素認識モデルの学習 (Phoneme Recognition Model Training)**

Experiment ID: 20251202_train_phoneme_recognition
Date: 2025-12-02
Author: [Terashima Kazuki]

## **1. 概要 (Overview)**

本実験は、**音素認識モデルの精度** を検証するために、**以下の3つのデータセット** を用いて **LSTMモデルによる学習** を行い、**正解率 (Accuracy) および 損失 (Loss)** を算出・可視化するものです。

* **ゴール:** 各データセットにおける音素認識精度の評価と、学習推移の確認。

## **2. 実験条件 (Conditions)**

使用するデータセットは以下の通りです。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :---- | :---- | :---- | :---- |
| **Nonpara30** | 通常発話データ | `dataset/preprocessed/nonpara30` | |
| **Nonpara30w** | 疑似ささやき声データ | `dataset/preprocessed/nonpara30w_ver2` | Ver2 |
| **Throat Mic** | ささやき声データ | `dataset/preprocessed/throat_microphone` | |

## **3. 手法詳細 (Methodology)**

### **LSTMモデルによる音素認識**

使用スクリプト: `scripts/train.py`

1. **Step 1: データ読み込み**
   * 指定されたデータセットからMSP (Mel-Spectrogram) 等の特徴量と音素ラベルを読み込む。
2. **Step 2: モデル学習**
   * LSTM (Long Short-Term Memory) ネットワークを用いて、入力特徴量から音素系列を予測するモデルを学習する。
   * 損失関数: CrossEntropyLoss
   * 最適化手法: AdamW
3. **Step 3: 評価・可視化**
   * テストデータに対する正解率を算出。
   * MSPと予測結果のヒートマップを表示・保存する。

## **4. 実験設定 (Configuration)**

`scripts/train.py` 内の `create_config` 関数にて設定。
詳細な設定は `config.yaml` ファイルに記述されており、以下の項目が含まれます。

```yaml
# config.yaml (抜粋)
sampling_rate: 22050
model_name: LSTM_net
hyperparameters:
  epochs: 3000
  batch_size: 128
  learning_rate: 2e-4
  frame_length: 88
  input_type: ['msp']
  output_type: ['ppgmat'] # Phoneme Posterior Gram
  seed: 1234
  jvs: true
  pseudo: true
  whisper: true
```

## **5. ディレクトリ構成 (Directory Structure)**

```text
20251202_train_phoneme_recognition/
├── README.md               # 本ドキュメント
├── config.yaml             # 実験設定ファイル
├── scripts/                # 実験用スクリプト
│   ├── train.py            # 学習スクリプト
│   ├── evaluate.py         # 評価スクリプト
│   ├── dataset.py          # データセット定義
│   └── prepare_dataset.py  # データセット準備スクリプト
├── model/                  # モデル定義など
│   └── lstm_net_rev.py     # LSTMモデル定義
├── dataset/                # データセットリスト
│   ├── train.txt           # 学習用リスト
│   └── test.txt            # 評価用リスト
├── trained_models/         # 学習済みモデルの出力ディレクトリ
│   └── {experiment_id}/    # 各実験のID (例: 20251203_101037)
│       ├── log.csv         # 学習ログ
│       ├── model.pth       # ベストモデル
│       ├── checkpoint.pth  # チェックポイント
│       └── config.yaml     # 学習時の設定ファイル
└── evaluation_results/     # 評価結果出力ディレクトリ
    └── {model_id}_{timestamp}/
        ├── accuracy.txt    # 精度結果
        └── heatmap_*.png   # ヒートマップ画像
```

## **6. 実行手順 (Usage)**

### **1. データセットの準備**

プロジェクトルート (`./master`) から以下のコマンドを実行し、学習・評価用リストを作成します。

```bash
python 20251202_train_phoneme_recognition/scripts/prepare_dataset.py
```

### **2. 学習の実行**

```bash
python 20251202_train_phoneme_recognition/scripts/train.py
```

### **3. 評価の実行**

学習済みモデルを用いて評価を行います。最新のモデルが自動的に選択されます。

```bash
python 20251202_train_phoneme_recognition/scripts/evaluate.py
```

評価結果は `evaluation_results/` ディレクトリに保存されます。

## **7. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

学習完了後、`result/log.csv` に記録されます。

### **考察 (Discussion)**

* **結果:** (実験後に記入)
* **課題:** (実験後に記入)
* **Next Step:** (実験後に記入)
