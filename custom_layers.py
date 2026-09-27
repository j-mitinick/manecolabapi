"""Camada de normalização serializada no treino (package='cxr').

Cópia funcional de PreprocessBackbone em treino_multiclasse_cnn.py.
Tem de ser importada ANTES de keras.models.load_model.
"""
from __future__ import annotations

from typing import Any

import keras
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable(package="cxr")
class PreprocessBackbone(layers.Layer):
    """Normalização específica de cada arquitectura (proibido reutilizar a mesma)."""

    def __init__(self, backend_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.backend_name = backend_name

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        nome = self.backend_name
        if nome == "resnet_v2":
            return keras.applications.resnet_v2.preprocess_input(x)
        if nome == "densenet":
            return keras.applications.densenet.preprocess_input(x)
        if nome == "efficientnet":
            return keras.applications.efficientnet.preprocess_input(x)
        if nome == "vgg16":
            return keras.applications.vgg16.preprocess_input(x)
        if nome == "mobilenet_v3":
            return keras.applications.mobilenet_v3.preprocess_input(x)
        return x / 255.0

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "backend_name": self.backend_name}
