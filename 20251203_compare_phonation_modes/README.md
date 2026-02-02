# **Phonation Mode Comparison**

Experiment ID: 20251203_compare_phonation_modes  
Date: 2025-12-04  
Author: Terashima

## **1. 概要 (Overview)**

本実験は、**VITSによる疑似ささやき声変換の精度** を検証するために、声質変換後の音声が目標とする疑似ささやき声にどれだけ近づいたのか定量的に評価する。

- **主な検証仮説:** VITSによる声質変換で、ささやき声が疑似ささやき声に変換されている
- **ゴール:** 疑似ささやき声への近さを定量的な指標 (MCD, Mel-MSE) で評価する

## **2. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :--- | :--- | :--- | :--- |
| **Target (目標)** | ターゲットの疑似ささやき声 | `dataset/nonpara30w_ver2` | 目標とする声質 |
| **Gen (生成)** | VITSによる変換後の疑似ささやき声 | `dataset/whisper_converted_v2` | 検証対象 |
| **Ref (参照)** | ベースラインのささやき声 | `dataset/whisper10` | 元のささやき声 |

## **3. 手法詳細 (Methodology)**

### **音響解析**

1. **Step 1:** DTW (Dynamic Time Warping) を用いてアライメントをとる
2. **Step 2:** Mel-MSE と MCD を算出する
3. **Step 3:** CSVに出力する

### **可視化**

- メルスペクトログラムを比較して視覚的に確認

## **4. 実験設定 (Configuration)**

| パラメータ | 値 |
| :--- | :--- |
| Sampling Rate | 22050 Hz |
| n_fft | 1024 |
| hop_length | 256 |
| n_mels | 80 |
| fmin | 0.0 Hz |
| fmax | 8000.0 Hz |

**評価指標:**
- Mel-MSE (Mel-Spectrogram Mean Squared Error)
- MCD (Mel-Cepstrum Distortion)

## **5. ディレクトリ構成 (Directory Structure)**

```text
20251203_compare_phonation_modes/
├── README.md                     # 本ドキュメント
├── scripts/                      # 実験用スクリプト
│   ├── calc_metrics.py           # 評価指標計算スクリプト
│   ├── visualize_mel.py          # メルスペクトログラム可視化スクリプト
│   ├── functions.py              # 共通関数
│   └── config.py                 # 設定ファイル
├── dataset/                      # 入力データ
│   ├── nonpara30w_ver2/          # ターゲット (疑似ささやき声)
│   ├── whisper_converted_v2/     # 生成 (VITS変換後)
│   └── whisper10/                # 参照 (ささやき声)
└── results/                      # 出力結果
    ├── metrics.csv               # 評価結果
    └── mel_spectrogram_*.png     # メルスペクトログラム画像
```

## **6. 実行手順 (Usage)**

### **6.1 評価指標の計算**

```bash
python scripts/calc_metrics.py \
    --gen_dir dataset/whisper_converted_v2 \
    --target_dir dataset/nonpara30w_ver2 \
    --ref_dir dataset/whisper10 \
    --output_dir results
```

### **6.2 メルスペクトログラムの可視化**

```bash
python scripts/visualize_mel.py \
    --gen_dir dataset/whisper_converted_v2 \
    --target_dir dataset/nonpara30w_ver2 \
    --ref_dir dataset/whisper10 \
    --output_dir results
```

## **7. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

| Metric | Ref vs Target (Mean ± Std) | Gen vs Target (Mean ± Std) |
| :--- | :--- | :--- |
| MCD | 11.7581 ± 0.6490 | 9.8764 ± 0.4452 |
| Mel_MSE | 1.9312 ± 0.4425 | 1.4642 ± 0.3501 |

### **定性評価 (Qualitative)**

メルスペクトログラム比較画像: `results/mel_spectrogram_*.png`

### **考察 (Discussion)**

- **結果:**
vitsを用いて疑似ささやき声を生成した結果、目標とする疑似ささやき声に比べて、MCDとMel-MSEの値が減少していることが確認できました。VITSを用いることで入力音声の音響特徴量が目標データに近似できた。
- **課題:**
- **Next Step:**
