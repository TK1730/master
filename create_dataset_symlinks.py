import os
import sys
import platform
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def create_symlink(source: Path, link_name: Path, target_is_directory=True):
    """
    プラットフォームに依存しない方法でシンボリックリンクを作成する

    Args:
        source: リンク元（実体のデータフォルダ）
        link_name: リンク先（リンクを作成する場所）
        target_is_directory: ターゲットがディレクトリかどうか

    Returns:
        (成功フラグ, メッセージ) のタプル
    """
    source = source.resolve()
    link_name = link_name.absolute()

    if not source.exists():
        print(f"[Skip] Source does not exist: {source}")
        return False, f"Source does not exist: {source}"

    if link_name.exists():
        print(f"[Skip] Link already exists: {link_name}")
        return False, f"Link already exists: {link_name}"

    # 親ディレクトリ作成
    link_name.parent.mkdir(parents=True, exist_ok=True)

    # 相対パスを計算
    rel_source = os.path.relpath(source, link_name.parent)

    try:
        if platform.system() == 'Windows':
            # === Windowsの場合 ===
            # 相対パスでシンボリックリンクを作成
            # (Windows 10以降、開発者モードなら管理者権限不要で作成可能)
            os.symlink(
                rel_source,
                link_name,
                target_is_directory=target_is_directory
            )
            print(f"[Created Symlink] {link_name} -> {rel_source}")
            return True, f"Created Symlink: {link_name} -> {rel_source}"
        else:
            # === Linux / Macの場合 ===
            os.symlink(
                rel_source,
                link_name,
                target_is_directory=target_is_directory
            )
            print(f"[Created Symlink] {link_name} -> {rel_source}")
            return True, f"Created Symlink: {link_name} -> {rel_source}"

    except Exception as e:
        error_msg = f"Failed to create link: {link_name}\nReason: {e}"
        print(f"[Error] {error_msg}")
        if platform.system() == 'Windows':
            tip = (
                "Tip: Windowsで相対パスのシンボリックリンクを作るには\n"
                "管理者権限で実行するか、Windowsの設定で「開発者モード」を有効にしてください。\n"
                "(従来のジャンクションは絶対パスしかサポートしていないため、シンボリックリンクを使用しています)"
            )
            print(tip)
            error_msg += f"\n{tip}"
        return False, error_msg


class SymlinkCreatorGUI:
    """
    シンボリックリンクをGUIで作成するためのクラス
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Symlink Creator - データフォルダリンク作成")
        self.root.geometry("700x500")

        # ソースフォルダ選択
        source_frame = ttk.LabelFrame(
            root, text="リンク元（実体のデータフォルダ）", padding=10
        )
        source_frame.pack(fill="x", padx=10, pady=5)

        self.source_path = tk.StringVar()
        entry = ttk.Entry(
            source_frame, textvariable=self.source_path, width=60
        )
        entry.pack(side="left", padx=5)
        btn = ttk.Button(
            source_frame, text="フォルダ選択...",
            command=self.select_source
        )
        btn.pack(side="left")

        # ターゲットフォルダ選択
        target_frame = ttk.LabelFrame(
            root, text="リンク先（リンクを作成する場所）", padding=10
        )
        target_frame.pack(fill="x", padx=10, pady=5)

        self.target_path = tk.StringVar()
        entry = ttk.Entry(
            target_frame, textvariable=self.target_path, width=60
        )
        entry.pack(side="left", padx=5)
        btn = ttk.Button(
            target_frame, text="フォルダ選択...",
            command=self.select_target
        )
        btn.pack(side="left")

        # ボタン
        button_frame = ttk.Frame(root, padding=10)
        button_frame.pack(fill="x", padx=10, pady=5)

        btn = ttk.Button(
            button_frame, text="シンボリックリンク作成",
            command=self.create_link
        )
        btn.pack(side="left", padx=5)
        btn = ttk.Button(
            button_frame, text="クリア", command=self.clear_fields
        )
        btn.pack(side="left", padx=5)

        # ログ表示エリア
        log_frame = ttk.LabelFrame(root, text="ログ", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=10, width=80)
        self.log_text.pack(fill="both", expand=True)

        # スクロールバー
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def select_source(self):
        """ソースフォルダを選択"""
        folder = filedialog.askdirectory(
            title="リンク元フォルダを選択（実体のデータフォルダ）"
        )
        if folder:
            self.source_path.set(folder)
            self.log(f"リンク元に選択: {folder}")

    def select_target(self):
        """ターゲットフォルダを選択"""
        folder = filedialog.askdirectory(
            title="リンク先フォルダを選択（リンクを作成する場所）"
        )
        if folder:
            self.target_path.set(folder)
            self.log(f"リンク先に選択: {folder}")

    def create_link(self):
        """シンボリックリンクを作成"""
        source = self.source_path.get()
        target = self.target_path.get()

        if not source or not target:
            msg = "ソースとターゲットの両方を選択してください。"
            messagebox.showwarning("入力エラー", msg)
            return

        source_path = Path(source)
        target_path = Path(target)

        # ディレクトリかどうか判定
        is_dir = source_path.is_dir() if source_path.exists() else True

        self.log("\nシンボリックリンク作成を開始...")
        self.log(f"  リンク元: {source_path}")
        self.log(f"  リンク先: {target_path}")

        success, message = create_symlink(
            source_path, target_path, target_is_directory=is_dir
        )

        if success:
            self.log(f"✓ 成功: {message}")
            messagebox.showinfo(
                "成功", "シンボリックリンクが正常に作成されました。"
            )
        else:
            self.log(f"✗ エラー: {message}")
            err_msg = (
                f"シンボリックリンクの作成に失敗しました。\n\n{message}"
            )
            messagebox.showerror("エラー", err_msg)

    def clear_fields(self):
        """入力フィールドをクリア"""
        self.source_path.set("")
        self.target_path.set("")
        self.log("フィールドをクリアしました。")

    def log(self, message):
        """ログメッセージを表示"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)


def run_gui():
    """GUI モードを起動"""
    root = tk.Tk()
    SymlinkCreatorGUI(root)
    root.mainloop()


def main():
    """メイン関数（従来のコマンドラインモード）"""
    # プロジェクトルート (このスクリプトの親ディレクトリと仮定)
    project_root = Path(__file__).parent

    # === 設定: リンクを作りたいリスト ===
    # (リンクを作る場所, 実体の場所)
    links_to_create = [
        # 例: 実験フォルダA
        (
            project_root /
            "20251126_scatter_phoneme_similarity_whisper_vs_vits/data/ref",
            project_root / "dataset/preprocessed/jvs_ver1/whisper10"
        ),
        (
            project_root /
            "20251126_scatter_phoneme_similarity_whisper_vs_vits/data/gen",
            project_root / "dataset/whisper_using_vits"
        ),
        # 例: 実験フォルダB
        # (
        #     project_root / "20251203_new_experiment/data/voiced",
        #     project_root / "dataset/preprocessed/jvs_ver1/nonpara30"
        # ),
    ]

    print("--- Creating Symlinks ---")
    for link_path, source_path in links_to_create:
        # ディレクトリかどうか判定
        is_dir = source_path.is_dir() if source_path.exists() else True
        success, message = create_symlink(
            source_path, link_path, target_is_directory=is_dir
        )
        if not success:
            print(f"Error creating link: {message}")
    print("--- Done ---")


if __name__ == "__main__":
    # コマンドライン引数をチェック
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        run_gui()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cli":
        main()
    else:
        # 引数がない場合はデフォルトでGUIモードを起動
        msg1 = "GUIモードで起動します。"
        msg2 = "コマンドラインモードを使用する場合は --cli オプションを指定してください。"
        print(msg1 + msg2)
        print("Usage: python create_dataset_symlinks.py [--gui|--cli]")
        run_gui()
