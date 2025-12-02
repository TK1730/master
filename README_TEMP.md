# **\[実験タイトル\]**

Experiment ID: \[YYYYMMDD\_action\_target\]  
Date: 2025-XX-XX  
Author: \[Your Name\]

## **1\. 概要 (Overview)**

本実験は、**\[目的: 何を明らかにしたいか\]** を検証するために、**\[対象データ\]** を用いて **\[分析手法\]** を行い、**\[評価指標\]** を算出・可視化するものです。

* **主な検証仮説:** \[例: VITSで生成した音声は、本物よりも音素認識率が低いのではないか？\]  
* **ゴール:** \[例: 音素一致率と意味的類似度の相関図を作成し、外れ値の傾向を掴む\]

## **2\. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :---- | :---- | :---- | :---- |
| **Ref (正解)** | \[例: 本物のささやき声\] | ../dataset/.../whisper10 | Ground Truth |
| **Gen (生成)** | \[例: VITS生成音声\] | ../dataset/.../whisper10\_vits | 比較対象 |

## **3\. 手法詳細 (Methodology)**

### **\[分析・生成プロセスの名前\]**

使用スクリプト: scripts/\[script\_name\].py

1. **Step 1:** \[処理内容 A\]  
   * \[詳細なアルゴリズムやツール名\]  
2. **Step 2:** \[処理内容 B\]  
3. **Step 3:** \[処理内容 C\]

## **4\. 実験設定 (Configuration)**

utils/config.py またはスクリプト引数で指定する重要なパラメータ。

* **Sampling Rate:** \[22050\] Hz  
* **Model Checkpoint:** \[例: exp\_vits\_ver1\_step50k.pth\]  
* **Evaluation Metrics:**  
  * **Phoneme Error Rate (PER):** \[使用するASRモデル名, 例: Whisper Large v3\]  
  * **Semantic Similarity:** \[使用するEmbeddingモデル名, 例: CLAP / BERT\]

## **5\. ディレクトリ構成 (Directory Structure)**

```text
\[Experiment\_ID\]/  
├── README.md               \# 本ドキュメント  
├── scripts/                \# 実験用スクリプト  
│   └── \[script\_name\].py  
├── data/                   \# 入力データ (シンボリックリンク推奨)  
│   ├── ref/                \# 正解データ  
│   └── gen/                \# 生成データ  
└── results/                \# 出力結果  
    ├── \[graph\].png         \# 可視化画像  
    └── \[metrics\].csv       \# 数値データ
```

## **6\. 実行手順 (Usage)**

### **\[手順名: 散布図の作成など\]**

\# プロジェクトルートで実行  
python \[Experiment\_ID\]/scripts/\[script\_name\].py \\  
    \--ref\_dir \[path\_to\_ref\] \\  
    \--gen\_dir \[path\_to\_gen\] \\  
    \--output\_dir \[Experiment\_ID\]/results \\  
    \--config \[config\_path\]

## **7\. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

| Metric | Score (Ref) | Score (Gen) | Diff |
| :---- | :---- | :---- | :---- |
| Metric A | \- | \[0.XX\] | \- |
| Metric B | \- | \[0.XX\] | \- |

### **考察 (Discussion)**

* **結果:** \[例: 散布図を見ると、右上にデータが集中しており相関が見られる。\]  
* **課題:** \[例: ファイルID 005 だけ極端に認識率が低い。聴取したところノイズが乗っていた。\]  
* **Next Step:** \[例: ノイズ除去フィルタを適用して再実験する。\]