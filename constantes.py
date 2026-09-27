"""Constantes partilhadas da API CXR."""
from __future__ import annotations

CLASSES: tuple[str, ...] = ("covid", "pneumonia", "tuberculose", "normal")
MODELOS_TODOS: tuple[str, ...] = ("vgg16", "resnet50v2", "densenet121", "efficientnetb4")
MODELOS_ENSEMBLE_OFICIAL: tuple[str, ...] = ("resnet50v2", "densenet121", "efficientnetb4")
MODELO_XAI_DEFAULT = "resnet50v2"
SEED_PADRAO = 42
LIMIAR_INCERTEZA = 0.50
FUNCOES = ("medico", "radiologista", "admin")
NIVEIS_EXPERIENCIA = ("junior", "pleno", "senior")
PESO_EXPERIENCIA = {"junior": 0.25, "pleno": 0.70, "senior": 1.00}

NOTA_CLINICA = (
    "Resultado de classificação de imagem (softmax, 4 classes exclusivas), não um "
    "diagnóstico médico. Padrões radiológicos de COVID-19, pneumonia e tuberculose "
    "podem sobrepor-se. O corpus de treino tem coincidência classe–fonte e não tem "
    "split por paciente. Correlacionar com clínica e exames laboratoriais. Não usar "
    "como dispositivo autónomo."
)
NOTA_LIMIARES = (
    "Limiares one-vs-rest calibrados no conjunto de validação do estudo para "
    "sensibilidade de triagem. NÃO servem para decidir multi-rótulo em produção; "
    "a classe prevista continua a ser o argmax da softmax."
)
NOTA_ENSEMBLE_4 = (
    "Não é o ensemble do estudo; incluído só para comparação na app."
)
NOTA_PROTOCOLO_FEEDBACK = (
    "O feedback não actualiza pesos em tempo real. Cada rótulo proposto fica "
    "pendente até um administrador o aceitar. O peso no corpus futuro depende do "
    "nível de experiência (júnior 0.25, pleno 0.70, sénior 1.00). Treino posterior "
    "é offline, com conjunto revisto — nunca a partir de um único parecer."
)
NOTA_TREINO_ESTUDO = (
    "O corpus original do estudo não é servido por esta API (volume, licenças e "
    "privacidade). Aqui vê-se a ficha de cada CNN da semente 42, limiares de "
    "triagem e o corpus de feedback aceite para um re-treino offline futuro."
)
