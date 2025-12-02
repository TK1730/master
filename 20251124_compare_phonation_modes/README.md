# 実験: 有声・疑似ささやき声・ささやき声の音響比較

## 1. 概要

本実験は、作成した「疑似ささやき声」が「ささやき声」とどのくらい異なるかを検証するために、以下の3種類の音声データの音響特徴量を比較・分析するものです。

## 2.比較条件

以下の3種類のデータセットを比較対象とする

| ラベル | 説明 | データソース |
| :--- | :--- | :--- |
| **Whisper (ささやき声)** | 実際に収録された本物のささやき声（Ground Truth）| whisper10 |
| **Topline (再合成)** | Pyworldで分析・再合成したささやき声（理論値の限界値）| whisper10_resynth_pyworld |
| **nonpara30w_ver1 (疑似ささやき声)** | 学部時代の疑似ささやき声生成方法によって作成 | nonpara30w_ver1 |
| **nonpara30w_ver2 (疑似ささやき声)** | 院時代の疑似ささやき声生成方法によって作成 | nonpara30w_ver2 |

## 3. 手法詳細

### 疑似ささやき声の生成 (Pseudo Generation)

#### nonpara30w_ver1

`create_pseudo/00_voice2voiceless_pyworld.py`に基づきPyWorldボコーダを用いた信号処理で生成

#### nonpara30w_ver2

`create_pseudo/00_voice2voiceless_pyworld_librosa.py`に基づきPyWorldボコーダを用いた信号処理で生成

### Toplineの生成 (Analysis-Synthesis)

#### whisper10_resynth_pyworld

`scripts/crete_topline.py` に基づき、本物のささやき声をPyworld(`harvest`)で分析し、加工せずに再合成  
Pyworldボコーダ自体が持つ「ささやき声再合成時の劣化量」を測定するためのベースライン

##### 実行コマンド

```bash
python scripts/create_topline.py `
>> --input_dir ../dataset/preprocessed/jvs_ver1/whisper10/ `
>> --output_dir ../dataset/preprocessed/whisper10_resynth_pyworld `
>> --sr 22050
```

## 4. 実験設定

`utils/config.py`および生成スクリプトの設定値

* **Sampling Rate:** 22,050 Hz
* **FFT Size:** 1024
* **Hop Length:** 256
* **Vocoder:** Pyworld

## 5. ディレクトリ構成

本実験フォルダの構成は以下の通り

```text
20251124_compare_phonation_modes/
|---README.md                   # 本ドキュメント   
|---scripts/                    # 実験用スクリプト
|   |---analyze_features.py     # MCD, MSE計算
|   |---create_topline.py       # Toplineデータ作成
----results/                    # 結果保存
```

## 6. 実行手順

### 1. 特徴量の比較  (Analyze Features)：nonpara30w_ver1

生成した疑似ささやき声 (`nonpara30w_ver1`) と、正解データのささやき声 (`whisper10`) を比較し、MCDやMSEを算出します。

#### 実行コマンド: ver1 の評価

```bash
python 20251124_compare_phonation_modes/scripts/analyze_features.py `
    --ref_dir ./dataset/preprocessed/jvs_ver1/whisper10 `
    --gen_dir ./dataset/preprocessed/jvs_ver1/nonpara30w_ver1 `
    --output_csv 20251124_compare_phonation_modes/results/metrics_ver1.csv `
    --sr 22050
```

### 2. 特徴量の比較 (Analyze Features)：nonpara30w_ver2

生成した疑似ささやき声 (`nonpara30w_ver2`) と、正解データのささやき声 (`whisper10`) を比較し、MCDやMSEを算出します。

#### 実行コマンド: ver2 の評価

```bash
python 20251124_compare_phonation_modes/scripts/analyze_features.py `
    --ref_dir ./dataset/preprocessed/jvs_ver1/whisper10 `
    --gen_dir ./dataset/preprocessed/jvs_ver1/nonpara30w_ver2 `
    --output_csv 20251124_compare_phonation_modes/results/metrics_ver2.csv `
    --sr 22050
```

### 3. 特徴量の比較 (Analyze Features)：Topline

worldボコーダで再合成した音声 (`whisper10_resynth_pyworld`) と、正解データのささやき声 (`whisper10`) を比較し、MCDやMSEを算出します。

#### 実行コマンド: Topline の評価

```bash
python 20251124_compare_phonation_modes/scripts/analyze_features.py `
    --ref_dir ./dataset/preprocessed/jvs_ver1/whisper10 `
    --gen_dir ./dataset/preprocessed/whisper10_resynth_pyworld `
    --output_csv 20251124_compare_phonation_modes/results/metrics_topline.csv `
    --sr 22050
```
