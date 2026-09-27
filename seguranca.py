"""JWT, hashing de senhas e RBAC."""
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .bd import obter_sessao
from .modelos_bd import Utilizador

ALGORITMO = "HS256"
esquema_bearer = HTTPBearer(auto_error=False)


def _segredo() -> str:
    valor = os.getenv("JWT_SECRET", "").strip()
    if not valor or valor == "altere-este-segredo":
        raise RuntimeError("Defina JWT_SECRET no .env (cadeia longa e aleatória).")
    return valor


def exigir_config_jwt() -> None:
    _segredo()


def minutos_expiracao() -> int:
    return int(os.getenv("JWT_EXPIRE_MINUTES", "480"))


def hash_senha(senha: str) -> str:
    return bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verificar_senha(senha: str, senha_hash: str) -> bool:
    try:
        return bcrypt.checkpw(senha.encode("utf-8"), senha_hash.encode("utf-8"))
    except ValueError:
        return False


def criar_token(utilizador: Utilizador) -> str:
    agora = datetime.now(timezone.utc)
    carga = {
        "sub": str(utilizador.id),
        "email": utilizador.email,
        "funcao": utilizador.funcao,
        "ver": utilizador.token_versao,
        "iat": int(agora.timestamp()),
        "exp": int((agora + timedelta(minutes=minutos_expiracao())).timestamp()),
    }
    return jwt.encode(carga, _segredo(), algorithm=ALGORITMO)


def utilizador_atual(
    credenciais: HTTPAuthorizationCredentials | None = Depends(esquema_bearer),
    sessao: Session = Depends(obter_sessao),
) -> Utilizador:
    if credenciais is None or credenciais.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token em falta. Inicie sessão.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        carga = jwt.decode(credenciais.credentials, _segredo(), algorithms=[ALGORITMO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Sessão expirada. Inicie sessão de novo.") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido.") from None

    try:
        uid = int(carga.get("sub", ""))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Token inválido.") from None

    utilizador = sessao.get(Utilizador, uid)
    if utilizador is None:
        raise HTTPException(status_code=401, detail="Utilizador inexistente.")
    if utilizador.bloqueado:
        raise HTTPException(status_code=403, detail="Conta bloqueada.")
    if int(carga.get("ver", -1)) != utilizador.token_versao:
        raise HTTPException(status_code=401, detail="Sessão invalidada. Inicie sessão de novo.")
    return utilizador


def exigir_funcoes(*funcoes: str) -> Callable[..., Utilizador]:
    def dependencia(utilizador: Utilizador = Depends(utilizador_atual)) -> Utilizador:
        if utilizador.funcao not in funcoes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Sem permissão para este recurso.",
            )
        return utilizador

    return dependencia
