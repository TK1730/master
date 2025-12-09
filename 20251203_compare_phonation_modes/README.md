# **Phonation Mode Comparison**

Experiment ID: 20251203_compare_phonation_modes
Date: 2025-12-04
Author: [Your Name]

## **1. 概要 (Overview)**

本実験は、**vitsによる疑似ささやき声がどれだけ疑似ささやき声に近づいたのか** を検証するために、**vitsによる声質変換** を行い、**疑似ささやき声への近さ** を定量的に評価するものです。

*   **主な検証仮説:** vitsによる声質変換でささやき声が疑似ささやき声に声質変換されているのではないか
*   **ゴール:** 疑似ささやき声にどれだけ近づいたのか定量的な評価を行う

## **2. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :--- | :--- | :--- | :--- |
| **Target (目標)** | ターゲットの疑似ささやき声 | `20251203_compare_phonation_modes/data/nonpara30w_ver2` | 目標とする声質 |
| **Gen (生成)** | vitsによる疑似ささやき声 | `20251203_compare_phonation_modes/data/whisper_converted_v2` | 検証対象 |
| **Ref (参照)** | ベースラインとしてささやき声 | `20251203_compare_phonation_modes/data/whisper10` | 元のささやき声 |

## **3. 手法詳細 (Methodology)**

### **音響解析**

使用スクリプト: `scripts/[script_name].py` (TBD)

1.  **Step 1:** DTW (Dynamic Time Warping) を用いてアライメントをとる
2.  **Step 2:** Mel-MSE (Mel-Spectrogram Mean Squared Error) と MCD (Mel-Cepstrum Distortion) を算出する
3.  **Step 3:** CSVに出力する

## **4. 実験設定 (Configuration)**

*   **Sampling Rate:** 22050 Hz
*   **Evaluation Metrics:**
    *   Mel-MSE (Mel-Spectrogram Mean Squared Error)
    *   MCD (Mel-Cepstrum Distortion)

## **5. ディレクトリ構成 (Directory Structure)**

```text
20251203_compare_phonation_modes/
├── README.md               # 本ドキュメント
├── scripts/                # 実験用スクリプト
├── data/                   # 入力データ
│   └── whisper_converted_v2 # 生成データ
└── results/                # 出力結果
```

## **6. 実行手順 (Usage)**

### **[手順名]**

```bash
# プロジェクトルートで実行
python 20251203_compare_phonation_modes/scripts/calc_metrics.py \
    --gen_dir 20251203_compare_phonation_modes/data/whisper_converted_v2 \
    --target_dir 20251203_compare_phonation_modes/data/nonpara30w_ver2 \
    --ref_dir 20251203_compare_phonation_modes/data/whisper10 \
    --output_dir 20251203_compare_phonation_modes/results
```

## **7. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

| Metric | Score (Target) | Score (Gen) | Diff |
| :--- | :--- | :--- | :--- |
| Mel-MSE | - | - | - |
| MCD | - | - | - |

### **考察 (Discussion)**

*   **結果:**
*   **課題:**
*   **Next Step:**
