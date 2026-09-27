"""Grad-CAM alinhado a treino_multiclasse_cnn.py, com procura recursiva de Conv2D."""
from __future__ import annotations

import base64
from io import BytesIO

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import Image

HEATMAP_PESO_TREINO = 0.45


def _iterar_camadas(camada: layers.Layer):
    yield camada
    subcamadas = getattr(camada, "layers", None)
    if subcamadas is None:
        return
    for sub in subcamadas:
        yield from _iterar_camadas(sub)


def ultima_conv(modelo: keras.Model) -> layers.Layer:
    """Última Conv2D. Primeiro no nível do modelo (como o treino); depois em profundidade."""
    ultima = None
    for camada in modelo.layers:
        if isinstance(camada, layers.Conv2D):
            ultima = camada
    if ultima is not None:
        return ultima
    for camada in _iterar_camadas(modelo):
        if isinstance(camada, layers.Conv2D):
            ultima = camada
    if ultima is None:
        raise RuntimeError("Nenhuma Conv2D encontrada para Grad-CAM.")
    return ultima


def _modelo_gradiente(modelo: keras.Model, conv: layers.Layer) -> keras.Model:
    try:
        return keras.Model(modelo.inputs, [conv.output, modelo.output])
    except (ValueError, AttributeError) as exc:
        try:
            camada = modelo.get_layer(conv.name)
            return keras.Model(modelo.inputs, [camada.output, modelo.output])
        except Exception:
            raise RuntimeError(
                f"Grad-CAM: não foi possível ligar a convolução '{conv.name}' ao grafo do modelo."
            ) from exc


def gradcam(
    modelo: keras.Model,
    imagens: np.ndarray,
    class_idx: int | None = None,
) -> np.ndarray:
    """Mapas Grad-CAM para um batch de imagens uint8 0-255. Devolve [B, H, W] em [0, 1]."""
    conv = ultima_conv(modelo)
    grad_model = _modelo_gradiente(modelo, conv)
    imgs = tf.convert_to_tensor(imagens)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(imgs, training=False)
        if class_idx is None:
            class_idx_t = tf.argmax(preds, axis=1)
        else:
            class_idx_t = tf.fill([tf.shape(preds)[0]], int(class_idx))
        perda = tf.gather(preds, class_idx_t, axis=1, batch_dims=1)
    grads = tape.gradient(perda, conv_out)
    if grads is None:
        raise RuntimeError("Grad-CAM: gradientes nulos (camada convolucional fora do grafo).")
    pesos = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(pesos * conv_out, axis=-1)
    cam = tf.nn.relu(cam)
    cam_min = tf.reduce_min(cam, axis=(1, 2), keepdims=True)
    cam_max = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam = tf.image.resize(cam[..., None], [imagens.shape[1], imagens.shape[2]])
    return cam.numpy()[..., 0]


def sobreposicao(
    img: np.ndarray,
    cam: np.ndarray,
    heatmap_peso: float = HEATMAP_PESO_TREINO,
) -> np.ndarray:
    """Overlay JET. Default 0.45 heatmap + 0.55 imagem (treino). alpha=0.4 → heatmap_peso=0.4."""
    heatmap = np.uint8(255 * np.clip(cam, 0, 1))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    base = img.astype(np.float32)
    resto = 1.0 - float(heatmap_peso)
    out = float(heatmap_peso) * heatmap.astype(np.float32) + resto * base
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_para_base64(overlay_rgb: np.ndarray, formato: str = "PNG") -> str:
    buf = BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format=formato)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if formato.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def mapa_classe_base64(
    modelo: keras.Model,
    batch_uint8: np.ndarray,
    class_idx: int,
    heatmap_peso: float = HEATMAP_PESO_TREINO,
) -> str:
    cam = gradcam(modelo, batch_uint8, class_idx=class_idx)
    overlay = sobreposicao(batch_uint8[0], cam[0], heatmap_peso=heatmap_peso)
    return overlay_para_base64(overlay)
