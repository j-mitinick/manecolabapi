"""Modelos SQLAlchemy."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .bd import Base


def _agora() -> datetime:
    return datetime.now(timezone.utc)


class Utilizador(Base):
    __tablename__ = "utilizadores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    primeiro_nome: Mapped[str] = mapped_column(String(100), nullable=False)
    ultimo_nome: Mapped[str] = mapped_column(String(100), nullable=False)
    senha_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    funcao: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    nivel_experiencia: Mapped[str] = mapped_column(String(16), nullable=False, default="pleno")
    bloqueado: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    token_versao: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    criado_em: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_agora)
    criado_por_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("utilizadores.id"), nullable=True
    )

    criado_por: Mapped[Utilizador | None] = relationship(
        "Utilizador",
        remote_side=[id],
        foreign_keys=[criado_por_id],
    )


class Predicao(Base):
    __tablename__ = "predicoes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    utilizador_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("utilizadores.id"), nullable=False, index=True
    )
    nome_ficheiro: Mapped[str] = mapped_column(String(255), nullable=False)
    hash_imagem: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    caminho_imagem: Mapped[str] = mapped_column(String(512), nullable=False)
    classe_predita: Mapped[str] = mapped_column(String(32), nullable=False)
    confianca: Mapped[float] = mapped_column(Float, nullable=False)
    alerta_incerteza: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    probabilidades_json: Mapped[str] = mapped_column(Text, nullable=False)
    criado_em: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_agora)

    utilizador: Mapped[Utilizador] = relationship("Utilizador")
    feedbacks: Mapped[list[Feedback]] = relationship("Feedback", back_populates="predicao")


class Feedback(Base):
    __tablename__ = "feedbacks"
    __table_args__ = (
        UniqueConstraint("predicao_id", "utilizador_id", name="uq_feedback_user_pred"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    predicao_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("predicoes.id"), nullable=False, index=True
    )
    utilizador_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("utilizadores.id"), nullable=False, index=True
    )
    classe_proposta: Mapped[str] = mapped_column(String(32), nullable=False)
    concorda_com_modelo: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comentario: Mapped[str | None] = mapped_column(Text, nullable=True)
    estado: Mapped[str] = mapped_column(String(16), nullable=False, default="pendente", index=True)
    peso: Mapped[float] = mapped_column(Float, nullable=False)
    nivel_experiencia: Mapped[str] = mapped_column(String(16), nullable=False)
    criado_em: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_agora)
    revisto_por_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("utilizadores.id"), nullable=True
    )
    revisto_em: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    nota_revisao: Mapped[str | None] = mapped_column(Text, nullable=True)

    predicao: Mapped[Predicao] = relationship("Predicao", back_populates="feedbacks")
    utilizador: Mapped[Utilizador] = relationship("Utilizador", foreign_keys=[utilizador_id])
    revisto_por: Mapped[Utilizador | None] = relationship(
        "Utilizador", foreign_keys=[revisto_por_id]
    )
