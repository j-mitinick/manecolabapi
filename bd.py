"""SQLite local — utilizadores, predições e feedback."""
from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

_RAIZ = Path(__file__).resolve().parent
DADOS_DIR = Path(os.getenv("DADOS_DIR", str(_RAIZ / "dados"))).expanduser().resolve()
DADOS_DIR.mkdir(parents=True, exist_ok=True)
(DADOS_DIR / "predicoes").mkdir(parents=True, exist_ok=True)

BD_PATH = DADOS_DIR / "cxr.sqlite"
ENGINE = create_engine(
    f"sqlite:///{BD_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
SessaoLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)
Base = declarative_base()


def obter_sessao() -> Generator[Session, None, None]:
    sessao = SessaoLocal()
    try:
        yield sessao
    finally:
        sessao.close()


def inicializar_bd() -> None:
    from . import modelos_bd  # noqa: F401

    Base.metadata.create_all(bind=ENGINE)
