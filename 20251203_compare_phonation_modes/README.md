# **Phonation Mode Comparison**

Experiment ID: 20251203_compare_phonation_modes
Date: 2025-12-03
Author: [Your Name]

## **1. 概要 (Overview)**

本実験は、**VITSによる疑似ささやき声が目標の疑似ささやき声に近づいたのか** を検証するために、**VITS生成音声と目標音声** を用いて **音響解析** を行い、その類似性や特性を評価するものです。

*   **主な検証仮説:** VITSによって変換された疑似ささやき声は、目標とする疑似ささやき声の音響的特徴に近づいているか。
*   **ゴール:** VITS生成音声と目標音声の比較検証。

## **2. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :--- | :--- | :--- | :--- |
| **Gen (生成)** | VITSによる疑似ささやき声 | `dataset/converted_whisper2voice_v2` | 検証対象 |
| **Target (目標)** | 目標 疑似ささやき声 | `dataset/preprocessed/jvs_ver1/nonpara30w_ver2` | 目標とする声質 |
| **Ref (参照)** | ささやき声 | `dataset/preprocessed/jvs_ver1/whisper10` | 元のささやき声 |

## **3. 手法詳細 (Methodology)**

### **音響解析**

使用スクリプト: `scripts/[script_name].py` (TBD)

1.  **Step 1:** [処理内容 A]
2.  **Step 2:** [処理内容 B]

## **4. 実験設定 (Configuration)**

*   **Sampling Rate:** [Value] Hz
*   **Evaluation Metrics:**
    *   [Metric A]

## **5. ディレクトリ構成 (Directory Structure)**

```text
20251203_compare_phonation_modes/
├── README.md               # 本ドキュメント
├── scripts/                # 実験用スクリプト
└── results/                # 出力結果
```

## **6. 実行手順 (Usage)**

### **[手順名]**

```bash
# プロジェクトルートで実行
python 20251203_compare_phonation_modes/scripts/calc_metrics.py \
    --gen_dir dataset/converted_whisper2voice_v2 \
    --target_dir dataset/preprocessed/jvs_ver1/nonpara30w_ver2 \
    --ref_dir dataset/preprocessed/jvs_ver1/whisper10 \
    --output_dir 20251203_compare_phonation_modes/results
```

## **7. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

| Metric | Score (Target) | Score (Gen) | Diff |
| :--- | :--- | :--- | :--- |
| Metric A | - | - | - |

### **考察 (Discussion)**

*   **結果:**
*   **課題:**
*   **Next Step:**
