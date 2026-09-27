"""Leitura de CXR para o tensor de entrada do modelo (uint8 0-255, 380×380×3).

Não aplica preprocess_input: essa camada já está dentro do .keras.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_TAMANHO = 380


class ImagemIlegivel(ValueError):
    """Ficheiro que não é uma imagem descodificável."""


def imagem_para_batch(conteudo: bytes, tamanho: int = IMG_TAMANHO) -> np.ndarray:
    """Bytes JPEG/PNG → ndarray uint8 (1, H, W, 3) em 0-255."""
    try:
        img = Image.open(BytesIO(conteudo))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagemIlegivel("Ficheiro ilegível ou formato de imagem não suportado.") from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((tamanho, tamanho), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ImagemIlegivel("A imagem não ficou com 3 canais RGB.")
    return arr[None, ...]
