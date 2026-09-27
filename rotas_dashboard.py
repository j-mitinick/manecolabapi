"""Páginas HTML do consola (login + dashboard)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
router = APIRouter(tags=["dashboard"], include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def raiz(pedido: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=pedido,
        name="dashboard.html",
        context={},
    )
