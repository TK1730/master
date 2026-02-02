import argparse
import csv
import numpy as np
import librosa
import pyworld as pw
from pathlib import Path
from tqdm import tqdm
import sys

# utilsがルートディレクトリにあるため、パスを追加
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import functions
import config

def compute_mcep(wav, sr, n_mcep=24, frame_period=5.0):
    """
    PyWorldを使用してメルケプストラム (MCP) を計算します。
    analyze_features.py と同じ処理です。
    """
    wav = wav.astype(np.float64)
    # F0推定 (dioを使用)
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    mcep = pw.code_spectral_envelope(sp, sr, n_mcep)
    return mcep

def compute_log_mel(
    wav,
    sr,
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    n_mels=config.n_mels
):
    """
    Librosaを使って対数メルスペクトログラムを抽出する
    analyze_features.py と同じ処理です。
    """
    # 振幅スペクトル
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    sp, _ = librosa.magphase(D)

    # メルフィルタバンク
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=config.fmin,
        fmax=config.fmax if config.fmax else None
    )
    mel_sp = np.dot(mel_basis, sp)

    # 対数変換 (dB)
    log_mel = functions.dynamic_range_compression(mel_sp)
    return log_mel.T  # (Time, n_mels)

def calculate_mcd_with_dtw(ref_wav, gen_wav, sr, n_mcep=24):
    """
    DTWアライメントを用いたMCD (メルケプストラム歪) を計算します。
    """
    ref_mcep = compute_mcep(ref_wav, sr, n_mcep)
    gen_mcep = compute_mcep(gen_wav, sr, n_mcep)

    # パワー係数 (0次元目) を除去
    ref_mcep_nopower = ref_mcep[:, 1:]
    gen_mcep_nopower = gen_mcep[:, 1:]

    # DTWアライメント
    d, path = librosa.sequence.dtw(
        ref_mcep_nopower.T,
        gen_mcep_nopower.T,
        metric='euclidean'
    )

    # 距離計算
    dist_sum = 0.0
    for i, j in path:
        diff = ref_mcep_nopower[i] - gen_mcep_nopower[j]
        dist_sum += np.sqrt(np.sum(diff ** 2))

    mean_dist = dist_sum / len(path)

    # dBに変換
    k_mcd = (10 * np.sqrt(2)) / np.log(10)
    mcd_db = k_mcd * mean_dist

    return mcd_db

def calculate_mel_mse_with_dtw(ref_wav, gen_wav, sr):
    """
    DTWを用いてアライメントを取り、メルスペクトログラムのMSEを計算する
    analyze_features.py と同じ処理です。
    """
    # 1. 特徴量抽出 (Log Mel Spectrogram)
    ref_mel = compute_log_mel(ref_wav, sr)
    gen_mel = compute_log_mel(gen_wav, sr)

    # 2. DTWによるアライメント (メルスペクトル同士で最短経路を探索)
    d, path = librosa.sequence.dtw(ref_mel.T, gen_mel.T, metric='euclidean')

    # 3. MSE (Mean Squared Error) の計算
    sq_err_sum = 0.0
    for i, j in path:
        diff = ref_mel[i] - gen_mel[j]
        sq_err_sum += np.mean(diff ** 2)  # フレームごとのMSE

    mean_mse = sq_err_sum / len(path)

    return mean_mse

def main(args):
    target_dir = Path(args.target_dir) # Target (疑似ささやき声)
    gen_dir = Path(args.gen_dir)       # Gen (VITS変換後)
    ref_dir = Path(args.ref_dir)       # Ref (元のささやき声)
    output_dir = Path(args.output_dir)
    print(gen_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "metrics.csv"

    # 共通ファイルを探す
    gen_files = sorted(list(gen_dir.rglob("*.wav")))
    
    print(f"生成ファイル数: {len(gen_files)}")
    
    results = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'MCD_Gen_vs_Target', 'Mel_MSE_Gen_vs_Target', 'MCD_Ref_vs_Target', 'Mel_MSE_Ref_vs_Target'])
        
        for gen_path in tqdm(gen_files):
            rel_path = gen_path.relative_to(gen_dir)
            target_path = target_dir / rel_path
            ref_path = ref_dir / rel_path
            
            # 対応するファイルが存在するか確認
            if not target_path.exists():
                candidates = list(target_dir.rglob(gen_path.name))
                if candidates:
                    target_path = candidates[0]
                else:
                    continue
            
            if not ref_path.exists():
                candidates = list(ref_dir.rglob(gen_path.name))
                if candidates:
                    ref_path = candidates[0]
                else:
                    continue
            
            try:
                # 音声読み込み
                gen_wav, _ = librosa.load(gen_path, sr=config.sr)
                target_wav, _ = librosa.load(target_path, sr=config.sr)
                ref_wav, _ = librosa.load(ref_path, sr=config.sr)
                
                # 指標計算: Gen vs Target
                mcd_gen = calculate_mcd_with_dtw(target_wav, gen_wav, config.sr)
                mse_gen = calculate_mel_mse_with_dtw(target_wav, gen_wav, config.sr)
                
                # 指標計算: Ref vs Target (ベースライン)
                mcd_ref = calculate_mcd_with_dtw(target_wav, ref_wav, config.sr)
                mse_ref = calculate_mel_mse_with_dtw(target_wav, ref_wav, config.sr)
                
                writer.writerow([
                    gen_path.name, 
                    f"{mcd_gen:.4f}", f"{mse_gen:.4f}",
                    f"{mcd_ref:.4f}", f"{mse_ref:.4f}"
                ])
                
                results.append({
                    'mcd_gen': mcd_gen,
                    'mse_gen': mse_gen,
                    'mcd_ref': mcd_ref,
                    'mse_ref': mse_ref
                })
                
            except Exception as e:
                print(f"エラー: {gen_path.name} の処理中にエラーが発生しました: {e}")

    # サマリー表示
    if results:
        avg_mcd_gen = np.mean([r['mcd_gen'] for r in results])
        std_mcd_gen = np.std([r['mcd_gen'] for r in results])
        
        avg_mse_gen = np.mean([r['mse_gen'] for r in results])
        std_mse_gen = np.std([r['mse_gen'] for r in results])
        
        avg_mcd_ref = np.mean([r['mcd_ref'] for r in results])
        std_mcd_ref = np.std([r['mcd_ref'] for r in results])
        
        avg_mse_ref = np.mean([r['mse_ref'] for r in results])
        std_mse_ref = np.std([r['mse_ref'] for r in results])

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(['--- Summary Statistics ---'])
            writer.writerow(['Metric', 'Mean_Ref_vs_Target', 'Std_Ref_vs_Target', 'Mean_Gen_vs_Target', 'Std_Gen_vs_Target'])
            writer.writerow(['MCD', f"{avg_mcd_ref:.4f}", f"{std_mcd_ref:.4f}", f"{avg_mcd_gen:.4f}", f"{std_mcd_gen:.4f}"])
            writer.writerow(['Mel_MSE', f"{avg_mse_ref:.4f}", f"{std_mse_ref:.4f}", f"{avg_mse_gen:.4f}", f"{std_mse_gen:.4f}"])
        
        print("\n" + "="*80)
        print("集計結果 (Summary Results)")
        print("="*80)
        print(f"{'Metric':<15} | {'Target vs Ref (Mean ± Std)':<30} | {'Target vs Gen (Mean ± Std)':<30}")
        print("-" * 80)
        print(f"{'MCD':<15} | {avg_mcd_ref:.4f} ± {std_mcd_ref:.4f}{'':<12} | {avg_mcd_gen:.4f} ± {std_mcd_gen:.4f}")
        print(f"{'Mel_MSE':<15} | {avg_mse_ref:.4f} ± {std_mse_ref:.4f}{'':<12} | {avg_mse_gen:.4f} ± {std_mse_gen:.4f}")
        print("="*80)
        print(f"結果は {output_csv} に保存されました。")
    else:
        print("有効な結果が得られませんでした。パスやデータを確認してください。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音響解析: Mel-MSEとMCDを計算して比較します")
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音声 (VITS) のディレクトリ")
    parser.add_argument("--target_dir", type=str, required=True, help="ターゲット音声 (疑似ささやき声) のディレクトリ")
    parser.add_argument("--ref_dir", type=str, required=True, help="参照音声 (元のささやき声) のディレクトリ")
    parser.add_argument("--output_dir", type=str, default="results", help="出力ディレクトリ")
    # sr引数はconfigから読み込むため削除しても良いが、互換性のため残すか迷うところ。
    # しかしanalyze_features.pyではconfigを使っているので、ここではconfig優先にする。
    # parser.add_argument("--sr", type=int, default=22050, help="サンプリング周波数")
    
    args = parser.parse_args()
    main(args)
