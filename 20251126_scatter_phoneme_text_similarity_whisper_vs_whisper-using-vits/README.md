# **\[実験：音素一致率とテキスト類似度の散布図分析\]**

Experiment ID: \[20251126_scatter_phoneme_text_similarity_whisper_vs_whisper-using-vits\_scatter\_whisper10 and whisper10_using_vits\]  
Date: 2025-11-27
Author: \[Kazuki Terashima\]

## **1\. 概要 (Overview)**

本実験は、**音素一致率が向上すれば有声発話へ変換した音声のテキスト類似度の向上がみらるか** を検証するために、**ささやき声とvitsによる疑似ささやき声 を用いて **音素認識モデルによる音素認識とReazonSpeechによる文字起こし** を行い、**音素一致率とテキスト類似度の散布図** を算出・可視化するものです。

* **主な検証仮説:**
  * VITSで生成した音声は、本物よりも音素認識率が低いのではないか？
  * 音素認識と有声発話に変換した後の音声のテキスト類似度に相関があるのではないか？
* **ゴール:** 音素一致率と文章類似度の相関図を作成し、相関の有無を確認

## **2\. 実験条件 (Conditions)**

比較・分析対象となるデータセットの定義。

| ラベル (ID) | 説明 | データソース (Path) | 備考 |
| :---- | :---- | :---- | :---- |
| **Ref (正解)** | \[本物のささやき声\] | ../dataset/preprocessed/jvs_ver1/whisper10 | Ground Truth |
| **Gen (生成)** | \[VITS生成音声\] | ../dataset/whisper_using_vits/ | 比較対象 |

## **3\. 手法詳細 (Methodology)**

### **\[分析・生成プロセスの名前\]**

使用スクリプト: scripts/\[script\_name\].py

1. **Step 1: 音素認識(Phoneme Recognition)**
   * \[音素認識モデル: LSTM(Bidirectional)\]
     * モデル定義: ./model/lstm_net_rev.py
     * 重みパス：./model/phoneme_bidirectional/
     * 手法：MSPから音素インデックス列を推定
     * 補足: 学部時代に学習させたモデルのため、新しい疑似ささやき声には対応していない

2. **Step 2: 音素一致率の算出**
   * RefとGenの音素列同士を比較し、一致している割合（0.1～1.0）を算出

3. **Step 3: テキスト類似度の読み込み(Text Similarity)**
   * データソース: ./data/converted_whisper2voice_v2_transcription_text.csv
   * ReazonSpeech等で文字起こしされた結果の類似度（ratioカラム）を使用

4. **Step 4:** 散布図作成(Visualization)
   * X軸：音素一致率
   * Y軸：テキスト類似度
   * 相関係数（Pearson）を算出し、correlation.csvに記載

## **4\. 実験設定 (Configuration)**

utils/config.py またはスクリプト引数で指定する重要なパラメータ。

* **Sampling Rate:** `22050` Hz
* **Target Column:** `ratio`(CSVないの類似度カラム名)
* **Model: Bidirectional LSTM**

## **5\. ディレクトリ構成 (Directory Structure)**

```text
20251126_scatter_phoneme_text_similarity_.../  
├── README.md               \# 本ドキュメント  
├── scripts/                \# 実験用スクリプト  
│   └── scatter_plot.py  
├── data/                   \# 入力データ  
│   └── converted_whisper2voice_v2_transcription.csv  
├── model/  
│   └── phoneme_bidirectional/
└── results/                \# 出力結果  
     ├── scatter_phoneme_vs_text_sim.png  
     ├── scatter_data.csv  
     ├── correlation.csv  
     └── metrics_phoneme_match.csv
```

## **6\. 実行手順 (Usage)**

### 以下のコマンドで実験を実行しました。(PowerShell環境)

```bash
python scripts/scatter_plot.py `
>> --ref_dir "../dataset/preprocessed/jvs_ver1/whisper10" `
>> --gen_dir "../dataset/whisper_using_vits" `
>> --model_path "./model/phoneme_bidirectional/" `
>> --sim_csv "./data/converted_whisper2voice_v2_transcription.csv" `
>> --output_dir "./result" `
>> --sr 22050 `
>> --csv_val_col "ratio"
```

## **7\. 結果ログ (Results Log)**

### **定量評価 (Quantitative)**

* **相関係数 (Peason)：**

$$
0.36396128161327673
$$

* **サンプル数：**

$$
495
$$

* **音素一致率と文章類似度の散布図：**

![散布図(Alt Text)](./result/scatter_phoneme_vs_text_sim.png)

* 音素一致率が0.4以下の場合低いテキスト類似度となっているが、それ以上のデータではテキスト類似度がちらばっている
* 1次近似の最小二乗法は右肩上がりのトレンドを示している。
* ピアソン相関は0.364で弱い相関を示している

### **考察 (Discussion)**

* 音素一致率と文章類似度には弱い相関があり、音素一致率が上がれば精度が向上することが見込める。しかし、音素認識モデルの精度が0.8程なためささやき声と一致するからといって精度が向上するといった関係性までは考えられない。そもそもの正解テキストとの一致率もみてみないと音素の獲得ができていないとは言えないことがわかった。
  
* **課題:** ささやき声とVITSによる疑似ささやき声の一致率による解析では、言語情報が維持できていないということには今のところ関係性はうすい  
* **Next Step:** ささやき声のppgmat(正しい音素配列)で同じように散布図を生成した場合を観察してVITSによる疑似ささやき声が言語情報を獲得できていないのかを確認する

### 考察 (Discussion)

* **結果の解釈:**
  * 音素一致率と文章類似度の間には、**弱い正の相関 (r ≈ 0.36)** が確認された。これは、音響的な音素構造の再現度が高まれば、テキストとしての了解性も向上する傾向を示唆している。
  * しかし、相関は強くない。散布図を見ると、**「音素一致率は高いが、文章類似度が低い」** データが散見される。

* **要因分析 (Limitation):**
  * **評価器の信頼性:** 使用した音素認識モデル（LSTM）は有声発話で学習されており（精度約0.8）、**ささやき声に対しては認識精度が大幅に低下している**と推測される。
  * **Refの不確実性:** 比較対象の `Ref`（本物のささやき声）の音素列自体が誤認識を含んでいる可能性が高く、これを正解（Ground Truth）として一致率を算出することには限界があることが判明した。

* **課題 (Challenges):**
  * 現状の「認識結果同士の比較」では、VITSが言語情報を維持できているかを正確に断定するにはノイズが多すぎる。
  * ASR（ReazonSpeech）がテキスト復元に失敗している要因が、「音素の崩れ」なのか、「有声音/無声音の混同（Voicing error）」なのかを切り分ける必要がある。

* **Next Step:**
  * **正解ラベルとの比較:** 認識結果同士ではなく、**「正解の音素ラベル (ppgmat)」** を基準として散布図を作成し直す。これにより、評価器の誤差を片側（Genのみ）に限定して分析を行う。
  * **Topline計測:** 本物のささやき声（Whisper）に対しても同様に正解ラベルとの一致率を出し、**「そもそもささやき声はどこまで認識可能なのか」** のベースラインを確認する。
