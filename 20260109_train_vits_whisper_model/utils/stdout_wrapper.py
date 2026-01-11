import sys
import tempfile
from typing import TextIO
import os


class StdoutWrapper(TextIO):
    """
    Google Colabとローカル環境で`sys.stdout`を切り替えるためのラッパークラス
    """

    def __init__(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        )
        self.original_stdout = sys.stdout

    def write(self, message: str) -> int:
        result = self.temp_file.write(message)
        self.temp_file.flush()
        print(message, end="", file=self.original_stdout)
        return result

    def flush(self) -> None:
        self.temp_file.flush()

    def read(self, n: int = -1) -> str:
        self.temp_file.seek(0)
        return self.temp_file.read(n)

    def close(self) -> None:
        self.temp_file.close()

    def fileno(self) -> int:
        return self.temp_file.fileno()


def is_running_in_colab() -> bool:
    """Return True when running inside Google Colab.

    This uses multiple heuristics in order of reliability:
    1. direct import of `google.colab`
    2. common Colab-specific environment variables (TPU/GPU)
    3. presence of `google.colab` or any module name containing "colab" in
       ``sys.modules`` when running inside an IPython kernel.

    These checks are defensive and avoid raising if IPython isn't available.
    """
    # Prefer using importlib to check availability without importing the module
    try:
        import importlib.util

        if importlib.util.find_spec("google.colab") is not None:
            return True
    except Exception:
        # If importlib isn't available, fall through to other checks
        pass

    # Check environment variables commonly set in Colab runtimes
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_TPU_ADDR"):
        return True

    # If google.colab has been imported elsewhere it will appear in sys.modules
    if "google.colab" in sys.modules:
        return True

    # Any module with 'colab' in its name is a hint
    for k in list(sys.modules.keys()):
        if "colab" in k:
            return True

    # No strong evidence of Colab
    return False


# Set SAFE_STDOUT based on a robust detection helper
if is_running_in_colab():
    SAFE_STDOUT = StdoutWrapper()
else:
    SAFE_STDOUT = sys.stdout
