"""Esquemas Pydantic da API (softmax multiclasse, não multi-rótulo)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ClasseNome = Literal["covid", "pneumonia", "tuberculose", "normal"]
FuncaoNome = Literal["medico", "radiologista", "admin"]
NivelExperiencia = Literal["junior", "pleno", "senior"]
EstadoFeedback = Literal["pendente", "aceite", "rejeitado"]


def _email_ok(valor: str) -> str:
    texto = valor.strip().lower()
    if "@" not in texto or " " in texto:
        raise ValueError("Email inválido.")
    local, _, dominio = texto.partition("@")
    if not local or not dominio or "." not in dominio:
        raise ValueError("Email inválido.")
    return texto


class SoftmaxClasses(BaseModel):
    covid: float
    pneumonia: float
    tuberculose: float
    normal: float


class BlocoEnsemble(BaseModel):
    componentes: list[str]
    agregacao: Literal["media_aritmetica"]
    probabilidades: SoftmaxClasses
    classe_predita: ClasseNome
    confianca: float
    nota: str | None = None


class ModeloInfo(BaseModel):
    nome: str
    ficheiro: str
    carregado: bool
    ensemble_oficial: bool


class HealthResposta(BaseModel):
    estado: str
    formulacao: Literal["multiclasse_softmax"]
    classes: list[str]
    device: str
    tensorflow: str
    keras: str
    modelos_carregados: list[str]
    model_dir: str


class PredictResposta(BaseModel):
    formulacao: Literal["multiclasse_softmax"]
    classes: list[str]
    predicao_id: int
    probabilidades_individuais: dict[str, SoftmaxClasses]
    ensemble_oficial: BlocoEnsemble
    ensemble_4_cnn: BlocoEnsemble
    alerta_incerteza: bool
    xai_modelo: str | None = None
    visualizacao_xai: dict[str, str] = Field(default_factory=dict)
    limiares_triagem_validacao: dict[str, Any] | None = None
    nota_limiares: str | None = None
    nota_clinica: str
    protocolo_feedback: str
    tempos_inferencia_s: dict[str, float]


class LoginPedido(BaseModel):
    email: str
    senha: str = Field(min_length=1)

    @field_validator("email")
    @classmethod
    def _email(cls, valor: str) -> str:
        return _email_ok(valor)


class SenhaPropria(BaseModel):
    senha_atual: str = Field(min_length=1)
    senha_nova: str = Field(min_length=8, max_length=128)


class SenhaAdmin(BaseModel):
    senha_nova: str = Field(min_length=8, max_length=128)


class UtilizadorCriar(BaseModel):
    email: str
    primeiro_nome: str = Field(min_length=1, max_length=100)
    ultimo_nome: str = Field(min_length=1, max_length=100)
    senha: str = Field(min_length=8, max_length=128)
    funcao: FuncaoNome
    nivel_experiencia: NivelExperiencia = "pleno"

    @field_validator("email")
    @classmethod
    def _email(cls, valor: str) -> str:
        return _email_ok(valor)


class UtilizadorActualizar(BaseModel):
    primeiro_nome: str | None = Field(default=None, min_length=1, max_length=100)
    ultimo_nome: str | None = Field(default=None, min_length=1, max_length=100)
    funcao: FuncaoNome | None = None
    nivel_experiencia: NivelExperiencia | None = None
    bloqueado: bool | None = None


class UtilizadorPublico(BaseModel):
    id: int
    email: str
    primeiro_nome: str
    ultimo_nome: str
    funcao: FuncaoNome
    nivel_experiencia: NivelExperiencia
    bloqueado: bool
    criado_em: datetime
    criado_por_id: int | None
    criado_por_nome: str | None = None


class TokenResposta(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_s: int
    utilizador: UtilizadorPublico


class FeedbackCriar(BaseModel):
    classe_proposta: ClasseNome
    comentario: str | None = Field(default=None, max_length=2000)


class FeedbackRevisao(BaseModel):
    estado: Literal["aceite", "rejeitado"]
    nota_revisao: str | None = Field(default=None, max_length=2000)


class FeedbackPublico(BaseModel):
    id: int
    predicao_id: int
    utilizador_id: int
    utilizador_nome: str
    classe_proposta: ClasseNome
    classe_modelo: ClasseNome
    concorda_com_modelo: bool
    comentario: str | None
    estado: EstadoFeedback
    peso: float
    nivel_experiencia: NivelExperiencia
    criado_em: datetime
    revisto_por_nome: str | None = None
    revisto_em: datetime | None = None
    nota_revisao: str | None = None
    aviso_proteccao: str | None = None


class PredicaoResumo(BaseModel):
    id: int
    nome_ficheiro: str
    classe_predita: ClasseNome
    confianca: float
    alerta_incerteza: bool
    criado_em: datetime
    autor_nome: str
    n_feedbacks: int


class ModeloTreino(BaseModel):
    nome: str
    ficheiro: str
    tamanho_bytes: int | None
    seed: int
    carregado: bool
    ensemble_oficial: bool
    classes: list[str]
    entrada: str
    formulacao: Literal["multiclasse_softmax"]
    limiares_triagem_validacao: dict[str, Any] | None = None
    notas_estudo: str
    protocolo_feedback: str
    corpus_feedback_aceite: dict[str, int]


class CorpusAmostra(BaseModel):
    feedback_id: int
    predicao_id: int
    classe_modelo: ClasseNome
    classe_proposta: ClasseNome
    peso: float
    nivel_experiencia: NivelExperiencia
    radiologista: str
    aceite_por: str | None
    aceite_em: datetime | None
    hash_imagem: str


class CorpusExportacao(BaseModel):
    nota: str
    n_amostras: int
    peso_total: float
    amostras: list[CorpusAmostra]
