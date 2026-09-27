"""Predições guardadas, feedback de radiologistas e corpus revisto."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, joinedload

from .bd import DADOS_DIR, obter_sessao
from .constantes import (
    CLASSES,
    NOTA_PROTOCOLO_FEEDBACK,
    PESO_EXPERIENCIA,
)
from .esquema import (
    CorpusAmostra,
    CorpusExportacao,
    FeedbackCriar,
    FeedbackPublico,
    FeedbackRevisao,
    PredicaoResumo,
)
from .modelos_bd import Feedback, Predicao, Utilizador
from .seguranca import exigir_funcoes, utilizador_atual

router = APIRouter(tags=["feedback"])
_radio = exigir_funcoes("radiologista", "admin")
_admin = exigir_funcoes("admin")


def _nome(u: Utilizador | None) -> str | None:
    if u is None:
        return None
    return f"{u.primeiro_nome} {u.ultimo_nome}"


def _aviso(fb: Feedback) -> str | None:
    if fb.nivel_experiencia == "junior":
        return (
            "Parecer de radiologista júnior: peso 0.25 e nunca entra no corpus "
            "sem aceitação explícita de um administrador."
        )
    if fb.estado == "pendente":
        return "Pendente de revisão. Os pesos do modelo não são actualizados com este rótulo."
    return None


def _para_feedback(fb: Feedback) -> FeedbackPublico:
    return FeedbackPublico(
        id=fb.id,
        predicao_id=fb.predicao_id,
        utilizador_id=fb.utilizador_id,
        utilizador_nome=_nome(fb.utilizador) or "",
        classe_proposta=fb.classe_proposta,  # type: ignore[arg-type]
        classe_modelo=fb.predicao.classe_predita,  # type: ignore[arg-type]
        concorda_com_modelo=fb.concorda_com_modelo,
        comentario=fb.comentario,
        estado=fb.estado,  # type: ignore[arg-type]
        peso=fb.peso,
        nivel_experiencia=fb.nivel_experiencia,  # type: ignore[arg-type]
        criado_em=fb.criado_em,
        revisto_por_nome=_nome(fb.revisto_por),
        revisto_em=fb.revisto_em,
        nota_revisao=fb.nota_revisao,
        aviso_proteccao=_aviso(fb),
    )


def _pode_ver(pred: Predicao, utilizador: Utilizador) -> bool:
    return utilizador.funcao == "admin" or pred.utilizador_id == utilizador.id


@router.get("/predicoes", response_model=list[PredicaoResumo])
def listar_predicoes(
    sessao: Session = Depends(obter_sessao),
    utilizador: Utilizador = Depends(utilizador_atual),
) -> list[PredicaoResumo]:
    q = sessao.query(Predicao).options(
        joinedload(Predicao.utilizador),
        joinedload(Predicao.feedbacks),
    )
    if utilizador.funcao != "admin":
        q = q.filter(Predicao.utilizador_id == utilizador.id)
    linhas = q.order_by(Predicao.id.desc()).limit(200).all()
    return [
        PredicaoResumo(
            id=p.id,
            nome_ficheiro=p.nome_ficheiro,
            classe_predita=p.classe_predita,  # type: ignore[arg-type]
            confianca=p.confianca,
            alerta_incerteza=p.alerta_incerteza,
            criado_em=p.criado_em,
            autor_nome=_nome(p.utilizador) or "",
            n_feedbacks=len(p.feedbacks),
        )
        for p in linhas
    ]


_TIPOS_IMAGEM = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _caminho_seguro_imagem(armazenado: str) -> Path | None:
    raiz = DADOS_DIR.resolve()
    candidatos = [Path(armazenado), raiz / "predicoes" / Path(armazenado).name]
    for bruto in candidatos:
        try:
            caminho = bruto.resolve()
        except OSError:
            continue
        if caminho.is_file() and caminho.is_relative_to(raiz):
            return caminho
    return None


@router.get("/predicoes/{pid}/imagem")
def obter_imagem_predicao(
    pid: int,
    sessao: Session = Depends(obter_sessao),
    utilizador: Utilizador = Depends(utilizador_atual),
) -> FileResponse:
    pred = sessao.query(Predicao).filter(Predicao.id == pid).one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Predição não encontrada.")
    if not _pode_ver(pred, utilizador):
        raise HTTPException(status_code=403, detail="Sem acesso a esta predição.")
    caminho = _caminho_seguro_imagem(pred.caminho_imagem)
    if caminho is None:
        raise HTTPException(status_code=404, detail="Ficheiro da imagem em falta no disco.")
    return FileResponse(
        caminho,
        media_type=_TIPOS_IMAGEM.get(caminho.suffix.lower(), "application/octet-stream"),
        filename=pred.nome_ficheiro,
    )


@router.post("/predicoes/{pid}/feedback", response_model=FeedbackPublico, status_code=201)
def criar_feedback(
    pid: int,
    pedido: FeedbackCriar,
    sessao: Session = Depends(obter_sessao),
    utilizador: Utilizador = Depends(_radio),
) -> FeedbackPublico:
    pred = (
        sessao.query(Predicao)
        .options(joinedload(Predicao.utilizador))
        .filter(Predicao.id == pid)
        .one_or_none()
    )
    if pred is None:
        raise HTTPException(status_code=404, detail="Predição não encontrada.")
    if not _pode_ver(pred, utilizador):
        raise HTTPException(status_code=403, detail="Sem acesso a esta predição.")
    existente = (
        sessao.query(Feedback)
        .filter(Feedback.predicao_id == pid, Feedback.utilizador_id == utilizador.id)
        .one_or_none()
    )
    if existente is not None:
        raise HTTPException(status_code=409, detail="Já submeteu feedback para esta predição.")
    if pedido.classe_proposta not in CLASSES:
        raise HTTPException(status_code=400, detail="Classe inválida.")
    peso = PESO_EXPERIENCIA.get(utilizador.nivel_experiencia, 0.25)
    fb = Feedback(
        predicao_id=pred.id,
        utilizador_id=utilizador.id,
        classe_proposta=pedido.classe_proposta,
        concorda_com_modelo=pedido.classe_proposta == pred.classe_predita,
        comentario=(pedido.comentario or "").strip() or None,
        estado="pendente",
        peso=peso,
        nivel_experiencia=utilizador.nivel_experiencia,
    )
    sessao.add(fb)
    sessao.commit()
    fb = (
        sessao.query(Feedback)
        .options(
            joinedload(Feedback.utilizador),
            joinedload(Feedback.predicao),
            joinedload(Feedback.revisto_por),
        )
        .filter(Feedback.id == fb.id)
        .one()
    )
    return _para_feedback(fb)


@router.get("/feedback", response_model=list[FeedbackPublico])
def listar_feedback(
    sessao: Session = Depends(obter_sessao),
    utilizador: Utilizador = Depends(_radio),
) -> list[FeedbackPublico]:
    q = sessao.query(Feedback).options(
        joinedload(Feedback.utilizador),
        joinedload(Feedback.predicao),
        joinedload(Feedback.revisto_por),
    )
    if utilizador.funcao != "admin":
        q = q.filter(Feedback.utilizador_id == utilizador.id)
    linhas = q.order_by(Feedback.id.desc()).limit(300).all()
    return [_para_feedback(f) for f in linhas]


@router.post("/admin/feedback/{fid}/revisao", response_model=FeedbackPublico)
def rever_feedback(
    fid: int,
    pedido: FeedbackRevisao,
    sessao: Session = Depends(obter_sessao),
    admin: Utilizador = Depends(_admin),
) -> FeedbackPublico:
    fb = (
        sessao.query(Feedback)
        .options(
            joinedload(Feedback.utilizador),
            joinedload(Feedback.predicao),
            joinedload(Feedback.revisto_por),
        )
        .filter(Feedback.id == fid)
        .one_or_none()
    )
    if fb is None:
        raise HTTPException(status_code=404, detail="Feedback não encontrado.")
    fb.estado = pedido.estado
    fb.nota_revisao = (pedido.nota_revisao or "").strip() or None
    fb.revisto_por_id = admin.id
    fb.revisto_em = datetime.now(timezone.utc)
    sessao.add(fb)
    sessao.commit()
    sessao.refresh(fb)
    fb = (
        sessao.query(Feedback)
        .options(
            joinedload(Feedback.utilizador),
            joinedload(Feedback.predicao),
            joinedload(Feedback.revisto_por),
        )
        .filter(Feedback.id == fid)
        .one()
    )
    return _para_feedback(fb)


@router.get("/admin/corpus", response_model=CorpusExportacao)
def exportar_corpus(
    sessao: Session = Depends(obter_sessao),
    _: Utilizador = Depends(_admin),
) -> CorpusExportacao:
    linhas = (
        sessao.query(Feedback)
        .options(
            joinedload(Feedback.utilizador),
            joinedload(Feedback.predicao),
            joinedload(Feedback.revisto_por),
        )
        .filter(Feedback.estado == "aceite")
        .order_by(Feedback.id.asc())
        .all()
    )
    amostras = [
        CorpusAmostra(
            feedback_id=f.id,
            predicao_id=f.predicao_id,
            classe_modelo=f.predicao.classe_predita,  # type: ignore[arg-type]
            classe_proposta=f.classe_proposta,  # type: ignore[arg-type]
            peso=f.peso,
            nivel_experiencia=f.nivel_experiencia,  # type: ignore[arg-type]
            radiologista=_nome(f.utilizador) or "",
            aceite_por=_nome(f.revisto_por),
            aceite_em=f.revisto_em,
            hash_imagem=f.predicao.hash_imagem,
        )
        for f in linhas
    ]
    return CorpusExportacao(
        nota=NOTA_PROTOCOLO_FEEDBACK,
        n_amostras=len(amostras),
        peso_total=round(sum(a.peso for a in amostras), 4),
        amostras=amostras,
    )
