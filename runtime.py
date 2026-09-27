"""Configuração TensorFlow antes de qualquer outro import Keras/TF da API."""
from __future__ import annotations

import logging
import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _carregar_dotenv() -> None:
    if not _ENV_PATH.is_file():
        return
    for linha in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


_carregar_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf  # noqa: E402

LOGGER = logging.getLogger("api_cxr")


def configurar_dispositivo(device: str) -> str:
    """device: auto | cpu | gpu. Devolve o dispositivo efectivo."""
    pedido = (device or "auto").strip().lower()
    intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "4"))
    inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "1"))
    gpus = tf.config.list_physical_devices("GPU")
    if pedido == "cpu" or (pedido == "auto" and not gpus):
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
        except RuntimeError:
            pass
        LOGGER.info("Inferência em CPU (intra_op=%s inter_op=%s)", intra, inter)
        return "cpu"
    if pedido == "gpu" and not gpus:
        raise RuntimeError("DEVICE=gpu mas não há GPU visível para o TensorFlow.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError:
        pass
    LOGGER.info("Inferência em GPU: %s", gpus)
    return "gpu"
