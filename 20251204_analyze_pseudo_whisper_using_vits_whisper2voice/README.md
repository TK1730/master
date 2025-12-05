# **VITS疑似ささやき声を使用した有声発話変換の音響分析**

Experiment ID: 20251204_analyze_pseudo_whisper_using_vits_whisper2voice  
Date: 2025-12-04  
Author: Terashima

## **1. 概要 (Overview)**

本実験は、**VITSによる疑似ささやき声を有声発話に変換した音声が有声発話とどれだけ音響的に異なるかを検証する**ために、**複数のささやき声変換データ**を用いて**音響特徴量分析**を行い、**メルケプストラム歪み(MCD)やメルスペクトログラム距離(MSP)**を算出・可視化するものです。

* **主な検証仮説:** ささやき声より疑似ささやき声に近づけたささやき声の方が変換した後の精度が向上しているはず。  
* **ゴール:** 疑似ささやき声の有声発話変換精度を定量的に評価し、ささやき声変換との比較を通じて変換品質の差を明らかにする。

## **2. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :---- | :---- | :---- | :---- |
| **Target (正解)** | 有声発話 (Ground Truth) | `dataset/preprocessed/jvs_ver1/nonpara30` | 目標の正解データ |
| **VITS Gen (検証)** | VITSによる疑似ささやき声の有声発話変換 | `dataset/whisper2voice/whisper_converted_v2` | 検証データ |
| **Whisper Conv (比較)** | ささやき声の有声発話変換音声 | `dataset/whisper2voice/whisper2voice` | 比較データ (ベースライン) |

## **3. 手法詳細 (Methodology)**

### **音響特徴量分析プロセス**

使用スクリプト: `scripts/calc_metrics.py`

1. **Step 1: データ読み込みとアライメント**  
   * DTW (Dynamic Time Warping) を使用して正解データと生成データの時間軸を揃える  
   * 各音声ファイルペアに対してフレームレベルのアラインメントを実施  

2. **Step 2: 音響特徴量抽出**  
   * メルケプストラム係数 (MCEP) の抽出  
   * メルスペクトログラム (Mel-Spectrogram) の計算  

3. **Step 3: 距離尺度の計算**  
   * MCD (Mel-Cepstral Distortion): メルケプストラム歪み  
   * MSP (Mel-Spectral Distance): メルスペクトログラム距離  

4. **Step 4: 統計分析と可視化**  
   * 各データセット間の距離尺度の平均値・標準偏差を算出  
   * 箱ひげ図やヒストグラムによる分布の可視化  

## **4. 実験設定 (Configuration)**

`scripts/calc_metrics.py` で指定する重要なパラメータ。

* **Sampling Rate:** 22050 Hz  
* **Frame Length:** 1024 samples  
* **Frame Shift:** 256 samples  
* **メルケプストラム次数:** 24次元  
* **Evaluation Metrics:**  
  * **MCD (Mel-Cepstral Distortion):** メルケプストラム係数間のユークリッド距離  
  * **MSP (Mel-Spectral Distance):** メルスペクトログラム間の距離

## **5. ディレクトリ構成 (Directory Structure)**

```text
20251204_analyze_pseudo_whisper_using_vits_whisper2voice/  
├── README.md               # 本ドキュメント  
├── scripts/                # 実験用スクリプト  
│   └── calc_metrics.py     # 音響特徴量分析スクリプト  
├── data/                   # 入力データ (シンボリックリンク推奨)  
│   ├── target/             # 正解データ (有声発話)  
│   ├── vits_gen/           # VITS疑似ささやき声変換  
│   └── whisper_conv/       # ささやき声変換 (ベースライン)  
└── results/                # 出力結果  
    ├── metrics.csv         # 数値データ (MCD, MSP)  
    ├── mcd_boxplot.png     # MCD の箱ひげ図  
    └── msp_histogram.png   # MSP のヒストグラム
```

## **6. 実行手順 (Usage)**

### **音響特徴量の計算と可視化**

```bash
# プロジェクトルートで実行  
python 20251204_analyze_pseudo_whisper_using_vits_whisper2voice/scripts/calc_metrics.py \
    --target_dir dataset/preprocessed/jvs_ver1/nonpara30 \
    --vits_gen_dir dataset/whisper2voice/whisper_converted_v2 \
    --whisper_conv_dir dataset/whisper2voice/whisper2voice \
    --output_dir 20251204_analyze_pseudo_whisper_using_vits_whisper2voice/results
```

### **パラメータ説明**

* `--target_dir`: 正解データ (有声発話) のディレクトリパス  
* `--vits_gen_dir`: VITS疑似ささやき声変換データのディレクトリパス  
* `--whisper_conv_dir`: ささやき声変換データのディレクトリパス  
* `--output_dir`: 結果出力先ディレクトリパス

## **7. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

| Metric | Target vs VITS Gen | Target vs Whisper Conv | Diff |
| :---- | :---- | :---- | :---- |
| MCD (Mean ± Std) | [- dB] | [- dB] | [- dB] |
| MSP (Mean ± Std) | [- dB] | [- dB] | [- dB] |

### **考察 (Discussion)**

* **結果:** [実験実施後に記入]  
* **課題:** [実験実施後に記入]  
* **Next Step:** [実験実施後に記入]

---

## **8. 参考文献・関連実験**

* 関連実験: `20251203_compare_phonation_modes` - 発声モード間の音響比較  
* VITSモデル: [使用したモデルのバージョンや設定]  
* Whisper2Voice変換: [変換手法の詳細]
