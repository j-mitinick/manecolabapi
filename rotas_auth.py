"""Login, perfil e alteração da própria senha."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .bd import obter_sessao
from .esquema import LoginPedido, SenhaPropria, TokenResposta, UtilizadorPublico
from .modelos_bd import Utilizador
from .seguranca import (
    criar_token,
    hash_senha,
    minutos_expiracao,
    utilizador_atual,
    verificar_senha,
)

router = APIRouter(prefix="/auth", tags=["autenticação"])


def nome_criador(utilizador: Utilizador) -> str | None:
    if utilizador.criado_por is None:
        return "sistema" if utilizador.criado_por_id is None else None
    return f"{utilizador.criado_por.primeiro_nome} {utilizador.criado_por.ultimo_nome}"


def para_publico(utilizador: Utilizador) -> UtilizadorPublico:
    return UtilizadorPublico(
        id=utilizador.id,
        email=utilizador.email,
        primeiro_nome=utilizador.primeiro_nome,
        ultimo_nome=utilizador.ultimo_nome,
        funcao=utilizador.funcao,  # type: ignore[arg-type]
        nivel_experiencia=utilizador.nivel_experiencia,  # type: ignore[arg-type]
        bloqueado=utilizador.bloqueado,
        criado_em=utilizador.criado_em,
        criado_por_id=utilizador.criado_por_id,
        criado_por_nome=nome_criador(utilizador),
    )


def bootstrap_admin(sessao: Session) -> None:
    if sessao.query(Utilizador).count() > 0:
        return
    email = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "admin@cxr.local").strip().lower()
    senha = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
    if not senha:
        raise RuntimeError("BOOTSTRAP_ADMIN_PASSWORD em falta no .env (primeiro administrador).")
    admin = Utilizador(
        email=email,
        primeiro_nome=os.getenv("BOOTSTRAP_ADMIN_PRIMEIRO_NOME", "Admin"),
        ultimo_nome=os.getenv("BOOTSTRAP_ADMIN_ULTIMO_NOME", "Sistema"),
        senha_hash=hash_senha(senha),
        funcao="admin",
        nivel_experiencia="senior",
        criado_por_id=None,
    )
    sessao.add(admin)
    sessao.commit()


@router.post("/login", response_model=TokenResposta)
def login(pedido: LoginPedido, sessao: Session = Depends(obter_sessao)) -> TokenResposta:
    email = str(pedido.email).strip().lower()
    utilizador = sessao.query(Utilizador).filter(Utilizador.email == email).one_or_none()
    if utilizador is None or not verificar_senha(pedido.senha, utilizador.senha_hash):
        raise HTTPException(status_code=401, detail="Email ou senha incorrectos.")
    if utilizador.bloqueado:
        raise HTTPException(status_code=403, detail="Conta bloqueada.")
    token = criar_token(utilizador)
    return TokenResposta(
        access_token=token,
        expires_in_s=minutos_expiracao() * 60,
        utilizador=para_publico(utilizador),
    )


@router.get("/me", response_model=UtilizadorPublico)
def eu(utilizador: Utilizador = Depends(utilizador_atual)) -> UtilizadorPublico:
    return para_publico(utilizador)


@router.post("/senha", response_model=UtilizadorPublico)
def actualizar_senha(
    pedido: SenhaPropria,
    sessao: Session = Depends(obter_sessao),
    utilizador: Utilizador = Depends(utilizador_atual),
) -> UtilizadorPublico:
    if not verificar_senha(pedido.senha_atual, utilizador.senha_hash):
        raise HTTPException(status_code=400, detail="Senha actual incorrecta.")
    if pedido.senha_atual == pedido.senha_nova:
        raise HTTPException(status_code=400, detail="A nova senha tem de ser diferente.")
    utilizador.senha_hash = hash_senha(pedido.senha_nova)
    utilizador.token_versao += 1
    sessao.add(utilizador)
    sessao.commit()
    sessao.refresh(utilizador)
    return para_publico(utilizador)
