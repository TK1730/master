# **VITSによる疑似ささやき声から有声発話への変換検証**

Experiment ID: 20251205_create_whisper2voice  
Date: 2025-12-05  
Author: terashima

## **1. 概要 (Overview)**

本実験は、**VITSによる疑似ささやき声を有声発話に変換してささやき声を入力するより精度が上がるか** を検証するために、**whisper_converted_v2とwhisper10** を用いて **音声生成とメルスペクトログラム比較** を行い、**MSE (Mean Squared Error) とMel Cepstral Distortion** を算出・可視化するものです。

* **主な検証仮説:** VITS生成の疑似ささやき声を有声発話に変換することで、自然な有声音声を生成できる
* **ゴール:** 生成された有声発話音声とターゲット音声(nonpara30)のメルスペクトログラムのMSEとMel Cepstral Distortionを計算し、変換の品質を定量評価する

## **2. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :---- | :---- | :---- | :---- |
| **Target (ターゲット)** | 有声発話音声 | ../dataset/preprocessed/nonpara30 | Ground Truth |
| **Whisper_Converted_V2** | VITS生成の疑似ささやき声 | ../dataset/.../whisper_converted_v2 | 有声発話への変換対象 |
| **Whisper10 (比較用)** | 実際のささやき声 | ../dataset/preprocessed/whisper10 | 比較対象 |

## **3. 手法詳細 (Methodology)**

本実験では、2つのステップで音声生成と音響解析を行います。

### **Step 1: 有声発話への変換**

使用スクリプト: `scripts/speech_synthesis.py`

1. VITS疑似ささやき声の有声発話への変換
   * whisper_converted_v2およびwhisper10から有声発話音声を生成
   * 音響モデル（PPG、F0MAT、MCP、CAP）を使用

### **Step 2: DTWベースの音響解析**

使用スクリプト: `scripts/acoustic_analysis.py`

1. **DTWアライメント**: 動的時間伸縮(DTW)を使用して、生成音声とターゲット音声の時間的なずれを補正
2. **メルスペクトログラムの抽出**: 生成された有声発話とターゲット(nonpara30)のメルスペクトログラムを計算
3. **評価指標の算出**:
   * **MSE (Mean Squared Error)**: DTWアライメント後のメルスペクトログラム間の平均二乗誤差
   * **MCD (Mel Cepstral Distortion)**: DTWアライメント後のメルケプストラル歪み
   * 参考として、DTWなしのメトリクスも計算

## **4. 実験設定 (Configuration)**

utils/config.py またはスクリプト引数で指定する重要なパラメータ。

* **Sampling Rate:** 22050 Hz
* **Model Checkpoint:** [変換モデル情報]
* **Evaluation Metrics:**
  * **MSE (Mean Squared Error):** メルスペクトログラム間の平均二乗誤差
  * **MCD (Mel Cepstral Distortion):** メルケプストラル歪み

## **5. ディレクトリ構成 (Directory Structure)**

```text
20251205_create_whisper2voice/
├── README.md                  # 本ドキュメント
├── scripts/                   # 実験用スクリプト
│   ├── speech_synthesis.py    # 有声発話生成スクリプト
│   ├── acoustic_analysis.py   # 音響解析スクリプト（DTW使用）
│   └── create_whisper2voice.py
├── dataset/                   # 入力データ
│   ├── nonpara30/             # ターゲット音声(有声発話)
│   ├── whisper_converted_v2/  # VITS生成疑似ささやき声
│   └── whisper10/             # 実際のささやき声
└── results/                   # 出力結果
    ├── generated/             # 生成された有声音声
    │   ├── whisper_converted_v2/
    │   └── whisper10/
    ├── metrics_whisper_converted_v2_dtw.csv  # DTWメトリクス
    └── metrics_whisper10_dtw.csv             # DTWメトリクス
```

## **6. 実行手順 (Usage)**

### **Step 1: 有声発話の生成**

```bash
# プロジェクトルートで実行
python 20251205_create_whisper2voice/scripts/speech_synthesis.py
```

このスクリプトは以下を実行します：
- whisper_converted_v2とwhisper10から有声音声を生成
- 結果を`results/generated/`に保存

### **Step 2: DTWベースの音響解析**

```bash
# プロジェクトルートで実行
python 20251205_create_whisper2voice/scripts/acoustic_analysis.py
```

このスクリプトは以下を実行します：
- 生成された有声音声とターゲット音声(nonpara30)を比較
- DTWアライメントを適用してメトリクスを計算
- 結果をCSVファイルに保存

## **7. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

| Metric | whisper_converted_v2 | whisper10 | 備考 |
| :---- | :---- | :---- | :---- |
| MSE (DTW) | [結果] | [結果] | DTWアライメント後の平均二乗誤差 |
| MCD (DTW) | [結果] | [結果] | DTWアライメント後のメルケプストラル歪み |
| MSE (No DTW) | [結果] | [結果] | 参考：DTWなしの平均二乗誤差 |
| MCD (No DTW) | [結果] | [結果] | 参考：DTWなしのメルケプストラル歪み |

### **考察 (Discussion)**

* **結果:** [実験結果に基づいた考察を記載]
* **課題:** [実験中に発見された課題を記載]
* **Next Step:** [次のステップや改善案を記載]
