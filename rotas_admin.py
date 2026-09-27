"""Gestão de utilizadores — só administrador."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from .bd import obter_sessao
from .esquema import SenhaAdmin, UtilizadorActualizar, UtilizadorCriar, UtilizadorPublico
from .modelos_bd import Utilizador
from .rotas_auth import para_publico
from .seguranca import exigir_funcoes, hash_senha

router = APIRouter(prefix="/admin/utilizadores", tags=["administração"])
_admin = exigir_funcoes("admin")


def _contar_admins_activos(sessao: Session) -> int:
    return (
        sessao.query(Utilizador)
        .filter(Utilizador.funcao == "admin", Utilizador.bloqueado.is_(False))
        .count()
    )


@router.get("", response_model=list[UtilizadorPublico])
def listar(
    sessao: Session = Depends(obter_sessao),
    _: Utilizador = Depends(_admin),
) -> list[UtilizadorPublico]:
    linhas = (
        sessao.query(Utilizador)
        .options(joinedload(Utilizador.criado_por))
        .order_by(Utilizador.id.asc())
        .all()
    )
    return [para_publico(u) for u in linhas]


@router.post("", response_model=UtilizadorPublico, status_code=201)
def criar(
    pedido: UtilizadorCriar,
    sessao: Session = Depends(obter_sessao),
    admin: Utilizador = Depends(_admin),
) -> UtilizadorPublico:
    email = str(pedido.email).strip().lower()
    if sessao.query(Utilizador).filter(Utilizador.email == email).one_or_none():
        raise HTTPException(status_code=409, detail="Já existe um utilizador com este email.")
    nivel = pedido.nivel_experiencia
    if pedido.funcao == "admin":
        nivel = "senior"
    novo = Utilizador(
        email=email,
        primeiro_nome=pedido.primeiro_nome.strip(),
        ultimo_nome=pedido.ultimo_nome.strip(),
        senha_hash=hash_senha(pedido.senha),
        funcao=pedido.funcao,
        nivel_experiencia=nivel,
        criado_por_id=admin.id,
    )
    sessao.add(novo)
    sessao.commit()
    sessao.refresh(novo)
    novo = (
        sessao.query(Utilizador)
        .options(joinedload(Utilizador.criado_por))
        .filter(Utilizador.id == novo.id)
        .one()
    )
    return para_publico(novo)


@router.patch("/{uid}", response_model=UtilizadorPublico)
def actualizar(
    uid: int,
    pedido: UtilizadorActualizar,
    sessao: Session = Depends(obter_sessao),
    admin: Utilizador = Depends(_admin),
) -> UtilizadorPublico:
    alvo = (
        sessao.query(Utilizador)
        .options(joinedload(Utilizador.criado_por))
        .filter(Utilizador.id == uid)
        .one_or_none()
    )
    if alvo is None:
        raise HTTPException(status_code=404, detail="Utilizador não encontrado.")
    dados = pedido.model_dump(exclude_unset=True)
    novo_bloqueado = dados.get("bloqueado", alvo.bloqueado)
    nova_funcao = dados.get("funcao", alvo.funcao)
    if (novo_bloqueado or nova_funcao != "admin") and alvo.funcao == "admin":
        restantes = _contar_admins_activos(sessao)
        if alvo.bloqueado is False and restantes <= 1:
            raise HTTPException(
                status_code=400,
                detail="Não é possível bloquear ou despromover o último administrador activo.",
            )
    if uid == admin.id and novo_bloqueado:
        raise HTTPException(status_code=400, detail="Não pode bloquear a própria conta.")
    if "primeiro_nome" in dados and dados["primeiro_nome"]:
        alvo.primeiro_nome = dados["primeiro_nome"].strip()
    if "ultimo_nome" in dados and dados["ultimo_nome"]:
        alvo.ultimo_nome = dados["ultimo_nome"].strip()
    if "funcao" in dados and dados["funcao"]:
        alvo.funcao = dados["funcao"]
    if "nivel_experiencia" in dados and dados["nivel_experiencia"]:
        alvo.nivel_experiencia = dados["nivel_experiencia"]
    if alvo.funcao == "admin":
        alvo.nivel_experiencia = "senior"
    if "bloqueado" in dados and dados["bloqueado"] is not None:
        alvo.bloqueado = bool(dados["bloqueado"])
        if alvo.bloqueado:
            alvo.token_versao += 1
    sessao.add(alvo)
    sessao.commit()
    sessao.refresh(alvo)
    return para_publico(alvo)


@router.post("/{uid}/senha", response_model=UtilizadorPublico)
def definir_senha(
    uid: int,
    pedido: SenhaAdmin,
    sessao: Session = Depends(obter_sessao),
    _: Utilizador = Depends(_admin),
) -> UtilizadorPublico:
    alvo = (
        sessao.query(Utilizador)
        .options(joinedload(Utilizador.criado_por))
        .filter(Utilizador.id == uid)
        .one_or_none()
    )
    if alvo is None:
        raise HTTPException(status_code=404, detail="Utilizador não encontrado.")
    alvo.senha_hash = hash_senha(pedido.senha_nova)
    alvo.token_versao += 1
    sessao.add(alvo)
    sessao.commit()
    sessao.refresh(alvo)
    return para_publico(alvo)
