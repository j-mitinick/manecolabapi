# Maneco Lab API

API local **FastAPI** para classificação **multiclasse** de radiografia de tórax (CXR). Apoio à leitura (CAD): exactamente uma de quatro classes via softmax / argmax.

`covid` · `pneumonia` · `tuberculose` · `normal`

**Não é um diagnóstico médico.** Padrões de COVID-19, pneumonia e tuberculose podem sobrepor-se. O corpus de treino do estudo tem coincidência classe–fonte e não tem split por paciente. Correlacionar sempre com clínica e exames laboratoriais. Não usar como dispositivo autónomo.

Repositório: [j-mitinick/manecolabapi](https://github.com/j-mitinick/manecolabapi)

## Ensemble oficial

Média aritmética (pesos iguais) de **ResNet50V2 + DenseNet121 + EfficientNetB4** (semente 42). A VGG16 é reportada à parte. `ensemble_4_cnn` é só comparação — **não** é o protocolo do estudo.

A API envia a imagem em RGB 0–255 a 380×380. O `preprocess_input` de cada backbone já está no `.keras` (`PreprocessBackbone`). **Não** volte a normalizar no cliente.

## Requisitos

- Python 3.11+ (testado em 3.13)
- Quatro ficheiros `.keras` da semente 42, **fora** deste repositório (são grandes):

Colocar em `../modelos_salvos_multiclasse/` (raiz do estudo) ou no caminho de `MODEL_DIR`:

- `vgg16_seed42.keras`
- `resnet50v2_seed42.keras`
- `densenet121_seed42.keras`
- `efficientnetb4_seed42.keras`

Opcionais: `*_seed42_limiares.json` (metadado de triagem, **não** multi-rótulo).

Não usar `mobilenetv3_*` nem `cnn_rasa_*`. O `.keras` sem `_seed42` no nome **não** é o da semente 42.

## Arranque

Na **raiz do estudo** (`project/`), não dentro de `api/`:

```bash
python -m pip install -r api/requirements.txt
copy api\.env.example api\.env
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Em Unix: `cp api/.env.example api/.env`.

Edite `api/.env`: defina um `JWT_SECRET` longo e a senha do primeiro administrador.

| Variável | Função |
|---|---|
| `JWT_SECRET` | Assinatura dos tokens (obrigatório) |
| `JWT_EXPIRE_MINUTES` | Validade da sessão (default 480) |
| `BOOTSTRAP_ADMIN_EMAIL` / `BOOTSTRAP_ADMIN_PASSWORD` | Primeiro admin, só se a BD estiver vazia |
| `MODEL_DIR` | Pasta dos `.keras` |
| `DEVICE` | `auto` · `cpu` · `gpu` |
| `CORS_ORIGINS` | Em produção, restrinja (não use `*`) |

Consola web: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
App na LAN: `http://<IP-LAN>:8000` · emulador Android: `http://10.0.2.2:8000`

Parar: **Ctrl+C** no terminal do uvicorn.

GPU se o TensorFlow a vir; senão CPU.

## Autenticação e papéis

As contas **não** se criam na app móvel. Só o administrador regista utilizadores (dashboard ou API).

- Público: `GET /health`, `POST /auth/login`
- Restante: `Authorization: Bearer <access_token>`
- Papéis: `medico` · `radiologista` · `admin`
- Radiologista / admin: feedback de rótulo e ficha de treino
- Médico: inferência CAD, sem submeter rótulo de ouro
- Conta bloqueada não entra; mudança de senha invalida o JWT

Admin inicial (altere já): ver `BOOTSTRAP_*` no `.env.example`.

## Endpoints principais

| Método | Caminho | Quem | Função |
|---|---|---|---|
| GET | `/health` | Público | Estado, device, modelos |
| POST | `/auth/login` | Público | JWT + perfil |
| GET | `/auth/me` | Autenticado | Perfil |
| POST | `/auth/senha` | Autenticado | Nova senha própria |
| GET | `/modelos` | Autenticado | Lista dos CNN |
| GET | `/modelos/treino` | Radiologista, admin | Ficha / corpus aceite |
| POST | `/predict` | Autenticado | Multipart `file` |
| GET | `/predicoes` | Autenticado | Histórico (próprias; admin vê todas) |
| GET | `/predicoes/{id}/imagem` | Autenticado | JPEG/PNG original |
| POST | `/predicoes/{id}/feedback` | Radiologista, admin | Uma `classe_proposta` |
| GET | `/feedback` | Radiologista, admin | Pareceres |
| POST | `/admin/feedback/{id}/revisao` | Admin | Aceitar / rejeitar |
| GET | `/admin/corpus` | Admin | Manifesto do corpus aceite |
| GET/POST/PATCH | `/admin/utilizadores` | Admin | Contas |

`POST /predict` (query):

- `incluir_xai=true|false` (default `true`)
- `classes_xai=covid` (repetível; mapas extra, **não** implica classes simultâneas)
- `modelo_xai=resnet50v2` (default)

A classe prevista é o argmax de `ensemble_oficial`. `alerta_incerteza` é verdadeiro se a confiança for &lt; 0,50. `visualizacao_xai` devolve data URIs (`data:image/png;base64,...`). `predicao_id` identifica o caso para feedback e para a imagem.

Formulação: **softmax de 4 classes exclusivas**, não multi-rótulo. O feedback envia **uma** `classe_proposta`. Fica `pendente` até um admin aceitar. **Não** há treino online: os `.keras` não mudam com o parecer.

## Exemplo

```bash
# login
curl -s -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@cxr.local\",\"senha\":\"SUA_SENHA\"}"

# inferência (substitua TOKEN)
curl -s -X POST "http://127.0.0.1:8000/predict?incluir_xai=true" \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@/caminho/para/cxr.png"

# imagem guardada
curl -s -o cxr.jpeg \
  -H "Authorization: Bearer TOKEN" \
  http://127.0.0.1:8000/predicoes/1/imagem
```

## Dados locais (não versionados)

| Caminho | Conteúdo |
|---|---|
| `dados/cxr.sqlite` | Utilizadores, predições, feedback |
| `dados/predicoes/` | Imagens originais enviadas |
| `.env` | Segredos |

## Licença e uso

Código de dissertação / investigação. Uso clínico autónomo não autorizado. Trate imagens e pareceres como dados de saúde.
