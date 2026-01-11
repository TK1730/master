import socket
from typing import Any, cast

from utils.logger import logger
from nlp.japanese.pyopenjtalk_worker.worker_common import (
    RequestType,
    receive_data,
    send_data,
)

"""
pyopenjtalk worker サーバーに接続し、リクエストを送信して結果を受け取るためのクライアント
側の機能を提供する
"""


class WorkerClient:
    """pyopenjtalk worker client"""

    def __init__(self, port: int) -> None:
        """
        Args:
            port (int): 接続先のサーバーが待ち受けているポート番号
        """
        # TCP/IP ソケットを作成してサーバーに接続
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # timeout: 60 seconds
        sock.settimeout(60)
        sock.connect((socket.gethostname(), port))
        self.sock = sock

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.sock.close()

    def dispatch_pyopenjtalk(
        self,
        func: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """ワーカプロセスに pyopenjtalk の関数実行をリクエストし、結果を受け取る

        Args:
            func (str): 実行する関数名

        Returns:
            Any: 関数の実行結果
        """
        data = {
            "request-type": RequestType.PYOPENJTALK,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
        return response.get("return")

    def status(self) -> int:
        """ワーカサーバーの現在の状態を取得する"""
        data = {"request-type": RequestType.STATUS}
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
        return cast(int, response.get("client-count"))

    def quit_server(self) -> None:
        """ワーカサーバプロセスに終了を指示する"""
        data = {"request-type": RequestType.QUIT_SERVER}
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
