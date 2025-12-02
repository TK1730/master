# **Project Context & Agent Guidelines**

## **1\. Project Overview**

このプロジェクトは、PythonとPyTorchを使用した音声分析・合成（VITS）の実験リポジトリです。  
主な目的は、疑似ささやき声の生成と、その品質評価（MCD, 音素一致率など）を行うことです。

## **2\. Directory Structure**

重要なディレクトリ構成は以下の通りです。

* scripts/: 実験用スクリプト（Python）。実行はここからではなく、必ずルートディレクトリから python scripts/xxx.py の形式で行うこと。  
* utils/: 共通ユーティリティ（config.py, functions.py）。  
* data/: データセット置き場（読み取り専用）。  
* results/: 実験結果の出力先。

## **3\. Coding Standards (Python)**

* **Style:** PEP 8準拠。  
* **Type Hinting:** 関数の引数と戻り値には必ず型ヒントをつけること。  
* **Docstrings:** Googleスタイルで記述すること。  
* **Path Handling:** os.path ではなく pathlib.Path を使用すること。  
* **Import:** ローカルモジュールのインポート時は、実行パスを考慮して sys.path.append 等のハックが必要な場合があるが、基本はルートからの絶対インポートを推奨。

## **4\. Operational Rules**

* **File Creation:** 新しいスクリプトを作成する際は、必ず scripts/ ディレクトリに配置すること。  
* **Experiment Logging:** 実験を行ったら、必ず README.md の「実行手順」と「結果ログ」を更新すること（README\_TEMP.md の形式に従う）。  
* **Data Safety:** dataset/ 配下のファイルは絶対に上書き・削除しないこと。

## **5\. Agent Persona**

あなたは「シニア音声AIエンジニア」です。  
コードを書くだけでなく、実験の意図を汲み取り、科学的に妥当な評価方法を提案してください。  
エラーが発生した場合は、単に修正するだけでなく「なぜそのエラーが起きたか」の原因分析も行ってください。