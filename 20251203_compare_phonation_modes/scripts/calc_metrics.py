import argparse
import csv
import numpy as np
import librosa
import pyworld as pw
from pathlib import Path
from tqdm import tqdm
import sys

# 定数定義 (utils/config.py から移植)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 0
FMAX = 8000 # または None (サンプリング周波数/2)

def compute_mcep(wav, sr, n_mcep=24, frame_period=5.0):
    """
    PyWorldを使用してメルケプストラム (MCP) を計算します。
    """
    wav = wav.astype(np.float64)
    # F0推定 (dioを使用)
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    mcep = pw.code_spectral_envelope(sp, sr, n_mcep)
    return mcep

def compute_modulation_spectrum(wav, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    変調スペクトル (MSP) を計算します。
    
    1. 対数メルスペクトログラムを計算
    2. 各メルバンドについて時間軸方向にFFTを計算
    3. 振幅を取って変調スペクトルを得る
    4. メルバンド全体で平均して大域的な変調スペクトル (変調周波数 vs 振幅) を得る
    """
    # 1. 対数メルスペクトログラム
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    sp, _ = librosa.magphase(D)
    
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=FMIN,
        fmax=FMAX
    )
    mel_sp = np.dot(mel_basis, sp)
    log_mel = np.log(mel_sp + 1e-6) # (n_mels, Time)

    # 2. 時間軸方向のFFT
    # エンベロープのスペクトルを取得したい
    # 平均を引いてDC成分 (静的エネルギー) を除去
    log_mel_centered = log_mel - np.mean(log_mel, axis=1, keepdims=True)
    
    # 時間軸でのFFT
    msp = np.abs(np.fft.rfft(log_mel_centered, axis=1))
    
    # 長さで正規化
    msp = msp / log_mel.shape[1]
    
    # 3. メルバンド全体での平均
    # 結果は (変調周波数,)
    msp_avg = np.mean(msp, axis=0)
    
    return msp_avg

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
    # librosa.sequence.dtw は (Feature, Time) を期待する
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
    # k = 10 * sqrt(2) / ln(10) approx 6.14
    k_mcd = (10 * np.sqrt(2)) / np.log(10)
    mcd_db = k_mcd * mean_dist

    return mcd_db

def calculate_msp_distance(ref_wav, gen_wav, sr):
    """
    変調スペクトル間のユークリッド距離を計算します。
    MSPは大域的な統計量 (時間平均) なので、DTWは不要です。
    """
    ref_msp = compute_modulation_spectrum(ref_wav, sr)
    gen_msp = compute_modulation_spectrum(gen_wav, sr)
    
    # 比較のために長さを揃える (補間)
    target_len = 100 # 比較のための任意の解像度
    
    x_ref = np.linspace(0, 1, len(ref_msp))
    x_gen = np.linspace(0, 1, len(gen_msp))
    x_target = np.linspace(0, 1, target_len)
    
    ref_msp_interp = np.interp(x_target, x_ref, ref_msp)
    gen_msp_interp = np.interp(x_target, x_gen, gen_msp)
    
    # ユークリッド距離
    dist = np.sqrt(np.sum((ref_msp_interp - gen_msp_interp) ** 2))
    
    return dist

def main(args):
    target_dir = Path(args.target_dir) # Target (疑似ささやき声)
    gen_dir = Path(args.gen_dir)       # Gen (VITS変換後)
    ref_dir = Path(args.ref_dir)       # Ref (元のささやき声)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "metrics.csv"

    # 共通ファイルを探す
    gen_files = sorted(list(gen_dir.rglob("*.wav")))
    
    print(f"生成ファイル数: {len(gen_files)}")
    
    results = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'MCD_Gen_vs_Target', 'MSP_Dist_Gen_vs_Target', 'MCD_Ref_vs_Target', 'MSP_Dist_Ref_vs_Target'])
        
        for gen_path in tqdm(gen_files):
            rel_path = gen_path.relative_to(gen_dir)
            target_path = target_dir / rel_path
            ref_path = ref_dir / rel_path
            
            # 対応するファイルが存在するか確認
            if not target_path.exists():
                # 名前で検索してみる
                candidates = list(target_dir.rglob(gen_path.name))
                if candidates:
                    target_path = candidates[0]
                else:
                    # print(f"警告: Targetファイルが見つかりません: {gen_path.name}")
                    continue
            
            if not ref_path.exists():
                candidates = list(ref_dir.rglob(gen_path.name))
                if candidates:
                    ref_path = candidates[0]
                else:
                    # print(f"警告: Refファイルが見つかりません: {gen_path.name}")
                    continue
            
            try:
                # 音声読み込み
                gen_wav, _ = librosa.load(gen_path, sr=args.sr)
                target_wav, _ = librosa.load(target_path, sr=args.sr)
                ref_wav, _ = librosa.load(ref_path, sr=args.sr)
                
                # 指標計算: Gen vs Target
                mcd_gen = calculate_mcd_with_dtw(target_wav, gen_wav, args.sr)
                msp_dist_gen = calculate_msp_distance(target_wav, gen_wav, args.sr)
                
                # 指標計算: Ref vs Target (ベースライン)
                mcd_ref = calculate_mcd_with_dtw(target_wav, ref_wav, args.sr)
                msp_dist_ref = calculate_msp_distance(target_wav, ref_wav, args.sr)
                
                writer.writerow([
                    gen_path.name, 
                    f"{mcd_gen:.4f}", f"{msp_dist_gen:.4f}",
                    f"{mcd_ref:.4f}", f"{msp_dist_ref:.4f}"
                ])
                
                results.append({
                    'mcd_gen': mcd_gen,
                    'msp_gen': msp_dist_gen,
                    'mcd_ref': mcd_ref,
                    'msp_ref': msp_dist_ref
                })
                
            except Exception as e:
                print(f"エラー: {gen_path.name} の処理中にエラーが発生しました: {e}")

    # サマリー表示
    if results:
        avg_mcd_gen = np.mean([r['mcd_gen'] for r in results])
        std_mcd_gen = np.std([r['mcd_gen'] for r in results])
        
        avg_msp_gen = np.mean([r['msp_gen'] for r in results])
        std_msp_gen = np.std([r['msp_gen'] for r in results])
        
        avg_mcd_ref = np.mean([r['mcd_ref'] for r in results])
        std_mcd_ref = np.std([r['mcd_ref'] for r in results])
        
        avg_msp_ref = np.mean([r['msp_ref'] for r in results])
        std_msp_ref = np.std([r['msp_ref'] for r in results])

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([]) # 空行を挿入して区切りを明確にする
            writer.writerow(['--- Summary Statistics ---'])
            writer.writerow(['Metric', 'Mean_Ref_vs_Target', 'Std_Ref_vs_Target', 'Mean_Gen_vs_Target', 'Std_Gen_vs_Target'])
            writer.writerow(['MCD', f"{avg_mcd_ref:.4f}", f"{std_mcd_ref:.4f}", f"{avg_mcd_gen:.4f}", f"{std_mcd_gen:.4f}"])
            writer.writerow(['MSP_Dist', f"{avg_msp_ref:.4f}", f"{std_msp_ref:.4f}", f"{avg_msp_gen:.4f}", f"{std_msp_gen:.4f}"])
        
        print("\n" + "="*80)
        print("集計結果 (Summary Results)")
        print("="*80)
        print(f"{'Metric':<15} | {'Target vs Ref (Mean ± Std)':<30} | {'Target vs Gen (Mean ± Std)':<30}")
        print("-" * 80)
        print(f"{'MCP (MCD)':<15} | {avg_mcd_ref:.4f} ± {std_mcd_ref:.4f}{'':<12} | {avg_mcd_gen:.4f} ± {std_mcd_gen:.4f}")
        print(f"{'MSP (Dist)':<15} | {avg_msp_ref:.4f} ± {std_msp_ref:.4f}{'':<12} | {avg_msp_gen:.4f} ± {std_msp_gen:.4f}")
        print("="*80)
        print(f"結果は {output_csv} に保存されました。")
    else:
        print("有効な結果が得られませんでした。パスやデータを確認してください。")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音響解析: MSPとMCPを計算して比較します")
    parser.add_argument("--gen_dir", type=str, required=True, help="生成音声 (VITS) のディレクトリ")
    parser.add_argument("--target_dir", type=str, required=True, help="ターゲット音声 (疑似ささやき声) のディレクトリ")
    parser.add_argument("--ref_dir", type=str, required=True, help="参照音声 (元のささやき声) のディレクトリ")
    parser.add_argument("--output_dir", type=str, default="results", help="出力ディレクトリ")
    parser.add_argument("--sr", type=int, default=22050, help="サンプリング周波数")
    
    args = parser.parse_args()
    main(args)
