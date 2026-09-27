"""API local FastAPI — classificação MULTICLASSE de CXR (CAD, não diagnóstico)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .runtime import configurar_dispositivo

DEVICE_EFECTIVO = configurar_dispositivo(os.getenv("DEVICE", "auto"))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from . import custom_layers  # noqa: E402, F401  — regista PreprocessBackbone (package cxr)
from .bd import DADOS_DIR, SessaoLocal, inicializar_bd, obter_sessao
from .constantes import (
    CLASSES,
    LIMIAR_INCERTEZA,
    MODELO_XAI_DEFAULT,
    MODELOS_ENSEMBLE_OFICIAL,
    MODELOS_TODOS,
    NOTA_CLINICA,
    NOTA_ENSEMBLE_4,
    NOTA_LIMIARES,
    NOTA_PROTOCOLO_FEEDBACK,
    NOTA_TREINO_ESTUDO,
    SEED_PADRAO,
)
from .custom_layers import PreprocessBackbone
from .esquema import (
    BlocoEnsemble,
    HealthResposta,
    ModeloInfo,
    ModeloTreino,
    PredictResposta,
    SoftmaxClasses,
)
from .explicabilidade import HEATMAP_PESO_TREINO, mapa_classe_base64
from .modelos_bd import Feedback, Predicao, Utilizador
from .preprocessamento import IMG_TAMANHO, ImagemIlegivel, imagem_para_batch
from .rotas_admin import router as router_admin
from .rotas_auth import bootstrap_admin, router as router_auth
from .rotas_dashboard import router as router_dashboard
from .rotas_feedback import router as router_feedback
from .seguranca import exigir_config_jwt, exigir_funcoes, utilizador_atual

LOGGER = logging.getLogger("api_cxr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MODEL_DIR = Path(
    os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "modelos_salvos_multiclasse"))
).expanduser().resolve()
XAI_HEATMAP_PESO = float(os.getenv("XAI_HEATMAP_PESO", str(HEATMAP_PESO_TREINO)))
_ESTATICO = Path(__file__).resolve().parent / "static"
_radio = exigir_funcoes("radiologista", "admin")


def _ficheiro_modelo(nome: str) -> Path:
    return MODEL_DIR / f"{nome}_seed{SEED_PADRAO}.keras"


def _softmax_dict(vec: np.ndarray) -> dict[str, float]:
    return {c: float(vec[i]) for i, c in enumerate(CLASSES)}


def _bloco_ensemble(
    nomes: tuple[str, ...],
    matriz: np.ndarray,
    nota: str | None = None,
) -> BlocoEnsemble:
    media = np.mean(matriz, axis=0)
    idx = int(np.argmax(media))
    return BlocoEnsemble(
        componentes=list(nomes),
        agregacao="media_aritmetica",
        probabilidades=SoftmaxClasses(**_softmax_dict(media)),
        classe_predita=CLASSES[idx],  # type: ignore[arg-type]
        confianca=float(media[idx]),
        nota=nota,
    )


class Estado:
    def __init__(self) -> None:
        self.modelos: dict[str, keras.Model] = {}
        self.limiares: dict[str, dict] = {}
        self.device: str = DEVICE_EFECTIVO

    def carregar(self) -> None:
        em_falta = [n for n in MODELOS_TODOS if not _ficheiro_modelo(n).is_file()]
        if em_falta:
            detalhes = ", ".join(f"{n} ({_ficheiro_modelo(n)})" for n in em_falta)
            raise RuntimeError(
                "Modelos em falta em MODEL_DIR. Coloque os .keras da semente 42. "
                f"Em falta: {detalhes}"
            )
        for nome in MODELOS_TODOS:
            caminho = _ficheiro_modelo(nome)
            LOGGER.info("A carregar %s", caminho)
            try:
                self.modelos[nome] = keras.models.load_model(
                    caminho,
                    compile=False,
                    custom_objects={"PreprocessBackbone": PreprocessBackbone},
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Falha ao carregar {caminho}. Confirme que custom_layers.PreprocessBackbone "
                    f"(package='cxr') está registada. Erro: {exc}"
                ) from exc
            path_lim = MODEL_DIR / f"{nome}_seed{SEED_PADRAO}_limiares.json"
            if path_lim.is_file():
                self.limiares[nome] = json.loads(path_lim.read_text(encoding="utf-8"))
        LOGGER.info("Modelos prontos: %s", list(self.modelos))

    def inferir(self, batch: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        probs: dict[str, np.ndarray] = {}
        tempos: dict[str, float] = {}
        for nome, modelo in self.modelos.items():
            t0 = time.perf_counter()
            saida = modelo.predict(batch, verbose=0)
            vec = np.asarray(saida, dtype=np.float64).reshape(-1)
            if vec.shape[0] != len(CLASSES):
                raise RuntimeError(
                    f"{nome} devolveu shape {vec.shape}, esperado ({len(CLASSES)},)."
                )
            probs[nome] = vec
            dt = time.perf_counter() - t0
            tempos[nome] = dt
            idx = int(np.argmax(vec))
            LOGGER.info(
                "%s argmax=%s conf=%.4f tempo=%.3fs",
                nome,
                CLASSES[idx],
                float(vec[idx]),
                dt,
            )
        return probs, tempos


estado = Estado()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    exigir_config_jwt()
    inicializar_bd()
    with SessaoLocal() as sessao:
        bootstrap_admin(sessao)
    try:
        estado.carregar()
    except Exception:
        LOGGER.exception("Arranque abortado: falha a carregar modelos.")
        raise
    yield
    estado.modelos.clear()


app = FastAPI(
    title="API CXR multiclasse",
    description=(
        "Classificação softmax de radiografia de tórax em covid / pneumonia / "
        "tuberculose / normal. Apoio à leitura (CAD), não diagnóstico. "
        "Excepto /health e /auth/login, os endpoints exigem JWT (Bearer) e RBAC."
    ),
    version="1.1.0",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)

_origens = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origens,
    allow_credentials=False if _origens == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_auth)
app.include_router(router_admin)
app.include_router(router_feedback)
app.include_router(router_dashboard)
if _ESTATICO.is_dir():
    app.mount("/static", StaticFiles(directory=str(_ESTATICO)), name="static")


@app.get("/health", response_model=HealthResposta, tags=["sistema"])
def health() -> HealthResposta:
    return HealthResposta(
        estado="ok" if estado.modelos else "a_carregar",
        formulacao="multiclasse_softmax",
        classes=list(CLASSES),
        device=estado.device,
        tensorflow=tf.__version__,
        keras=keras.__version__,
        modelos_carregados=list(estado.modelos),
        model_dir=str(MODEL_DIR),
    )


@app.get("/modelos", response_model=list[ModeloInfo], tags=["modelos"])
def listar_modelos(_: Utilizador = Depends(utilizador_atual)) -> list[ModeloInfo]:
    return [
        ModeloInfo(
            nome=nome,
            ficheiro=str(_ficheiro_modelo(nome)),
            carregado=nome in estado.modelos,
            ensemble_oficial=nome in MODELOS_ENSEMBLE_OFICIAL,
        )
        for nome in MODELOS_TODOS
    ]


@app.get("/modelos/treino", response_model=list[ModeloTreino], tags=["modelos"])
def dados_treino(
    sessao: Session = Depends(obter_sessao),
    _: Utilizador = Depends(_radio),
) -> list[ModeloTreino]:
    contagens = {c: 0 for c in CLASSES}
    for classe, n in (
        sessao.query(Feedback.classe_proposta, func.count(Feedback.id))
        .filter(Feedback.estado == "aceite")
        .group_by(Feedback.classe_proposta)
        .all()
    ):
        if classe in contagens:
            contagens[classe] = int(n)
    fichas: list[ModeloTreino] = []
    for nome in MODELOS_TODOS:
        caminho = _ficheiro_modelo(nome)
        tamanho = caminho.stat().st_size if caminho.is_file() else None
        fichas.append(
            ModeloTreino(
                nome=nome,
                ficheiro=str(caminho),
                tamanho_bytes=tamanho,
                seed=SEED_PADRAO,
                carregado=nome in estado.modelos,
                ensemble_oficial=nome in MODELOS_ENSEMBLE_OFICIAL,
                classes=list(CLASSES),
                entrada=f"RGB 0–255, {IMG_TAMANHO}×{IMG_TAMANHO}",
                formulacao="multiclasse_softmax",
                limiares_triagem_validacao=estado.limiares.get(nome),
                notas_estudo=NOTA_TREINO_ESTUDO,
                protocolo_feedback=NOTA_PROTOCOLO_FEEDBACK,
                corpus_feedback_aceite=contagens,
            )
        )
    return fichas


def _classes_xai_pedidas(
    incluir_xai: bool,
    classes_xai: list[str] | None,
    classe_argmax: str,
) -> list[str]:
    if not incluir_xai:
        return []
    pedidas = [classe_argmax]
    for c in classes_xai or []:
        if c not in CLASSES:
            raise HTTPException(
                status_code=400,
                detail=f"classes_xai inválida: {c}. Use {list(CLASSES)}.",
            )
        if c not in pedidas:
            pedidas.append(c)
    return pedidas


def _gerar_xai(batch: np.ndarray, modelo_xai: str, classes_alvo: list[str]) -> dict[str, str]:
    mapas: dict[str, str] = {}
    for nome_cls in classes_alvo:
        idx = CLASSES.index(nome_cls)
        mapas[nome_cls] = mapa_classe_base64(
            estado.modelos[modelo_xai],
            batch,
            idx,
            heatmap_peso=XAI_HEATMAP_PESO,
        )
    return mapas


def _guardar_predicao(
    sessao: Session,
    utilizador: Utilizador,
    nome_ficheiro: str,
    bruto: bytes,
    ens_oficial: BlocoEnsemble,
    individuais: dict[str, SoftmaxClasses],
) -> int:
    digest = hashlib.sha256(bruto).hexdigest()
    ext = Path(nome_ficheiro or "cxr.png").suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        ext = ".png"
    destino = DADOS_DIR / "predicoes" / f"{uuid.uuid4().hex}{ext}"
    destino.write_bytes(bruto)
    rec = Predicao(
        utilizador_id=utilizador.id,
        nome_ficheiro=nome_ficheiro or destino.name,
        hash_imagem=digest,
        caminho_imagem=str(destino),
        classe_predita=ens_oficial.classe_predita,
        confianca=ens_oficial.confianca,
        alerta_incerteza=ens_oficial.confianca < LIMIAR_INCERTEZA,
        probabilidades_json=json.dumps(
            {
                "ensemble_oficial": ens_oficial.probabilidades.model_dump(),
                "individuais": {k: v.model_dump() for k, v in individuais.items()},
            },
            ensure_ascii=False,
        ),
    )
    sessao.add(rec)
    sessao.commit()
    sessao.refresh(rec)
    return rec.id


@app.post("/predict", response_model=PredictResposta, tags=["inferência"])
async def predict(
    file: UploadFile = File(..., description="Imagem JPEG/PNG de CXR"),
    incluir_xai: bool = Query(True),
    classes_xai: list[str] | None = Query(
        default=None,
        description="Classes extra para Grad-CAM (não implica presença simultânea).",
    ),
    modelo_xai: str = Query(MODELO_XAI_DEFAULT),
    sessao: Session = Depends(obter_sessao),
    utilizador: Utilizador = Depends(utilizador_atual),
) -> PredictResposta:
    if not estado.modelos:
        raise HTTPException(status_code=503, detail="Modelos ainda não carregados.")
    if modelo_xai not in estado.modelos:
        raise HTTPException(status_code=400, detail=f"modelo_xai inválido: {modelo_xai}.")

    bruto = await file.read()
    if not bruto:
        raise HTTPException(status_code=400, detail="Ficheiro vazio.")
    try:
        batch = imagem_para_batch(bruto, tamanho=IMG_TAMANHO)
    except ImagemIlegivel as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        probs, tempos = await run_in_threadpool(estado.inferir, batch)
    except Exception:
        LOGGER.exception("Falha na inferência")
        raise HTTPException(status_code=500, detail="Erro interno na inferência.") from None

    individuais = {n: SoftmaxClasses(**_softmax_dict(probs[n])) for n in MODELOS_TODOS}
    mat_oficial = np.stack([probs[n] for n in MODELOS_ENSEMBLE_OFICIAL], axis=0)
    mat_4 = np.stack([probs[n] for n in MODELOS_TODOS], axis=0)
    ens_oficial = _bloco_ensemble(MODELOS_ENSEMBLE_OFICIAL, mat_oficial)
    ens_4 = _bloco_ensemble(MODELOS_TODOS, mat_4, nota=NOTA_ENSEMBLE_4)

    LOGGER.info(
        "ensemble_oficial argmax=%s conf=%.4f user=%s",
        ens_oficial.classe_predita,
        ens_oficial.confianca,
        utilizador.email,
    )

    mapas: dict[str, str] = {}
    xai_usado: str | None = None
    if incluir_xai:
        xai_usado = modelo_xai
        alvo = _classes_xai_pedidas(True, classes_xai, ens_oficial.classe_predita)
        t0 = time.perf_counter()
        try:
            mapas = await run_in_threadpool(_gerar_xai, batch, modelo_xai, alvo)
        except Exception:
            LOGGER.exception("Falha no Grad-CAM")
            raise HTTPException(status_code=500, detail="Erro ao gerar Grad-CAM.") from None
        tempos["gradcam"] = time.perf_counter() - t0

    predicao_id = _guardar_predicao(
        sessao,
        utilizador,
        file.filename or "cxr.png",
        bruto,
        ens_oficial,
        individuais,
    )

    return PredictResposta(
        formulacao="multiclasse_softmax",
        classes=list(CLASSES),
        predicao_id=predicao_id,
        probabilidades_individuais=individuais,
        ensemble_oficial=ens_oficial,
        ensemble_4_cnn=ens_4,
        alerta_incerteza=ens_oficial.confianca < LIMIAR_INCERTEZA,
        xai_modelo=xai_usado,
        visualizacao_xai=mapas,
        limiares_triagem_validacao=estado.limiares or None,
        nota_limiares=NOTA_LIMIARES if estado.limiares else None,
        nota_clinica=NOTA_CLINICA,
        protocolo_feedback=NOTA_PROTOCOLO_FEEDBACK,
        tempos_inferencia_s={k: round(v, 4) for k, v in tempos.items()},
    )
