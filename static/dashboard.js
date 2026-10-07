const CHAVE_TOKEN = "cxr_jwt";
const CHAVE_USER = "cxr_user";
const TITULOS = {
  inicio: "Início",
  inferencia: "Ler",
  modelos: "Modelos",
  feedback: "Feedback",
  utilizadores: "Utilizadores",
  conta: "Perfil",
};
const PAPEIS = { medico: "Médico", radiologista: "Radiologista", admin: "Administrador" };
const NIVEIS = { junior: "Júnior", pleno: "Pleno", senior: "Sénior" };
const NOMES_CLASSE = {
  covid: "Covid-19",
  pneumonia: "Pneumonia",
  tuberculose: "Tuberculose",
  normal: "Normal",
};
const NOMES_MODELO = {
  ensemble: "Ensemble",
  vgg16: "VGG16",
  resnet50v2: "ResNet50V2",
  densenet121: "DenseNet121",
  efficientnetb4: "EfficientNetB4",
};
const ORDEM_MODELOS = ["ensemble", "vgg16", "resnet50v2", "densenet121", "efficientnetb4"];

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

let ultimaImagemUrl = "";
let ultimaResposta = null;

function utilizador() {
  try { return JSON.parse(localStorage.getItem(CHAVE_USER) || "null"); }
  catch { return null; }
}

function token() { return localStorage.getItem(CHAVE_TOKEN); }

function guardarSessao(data) {
  localStorage.setItem(CHAVE_TOKEN, data.access_token);
  localStorage.setItem(CHAVE_USER, JSON.stringify(data.utilizador));
}

function sair() {
  localStorage.removeItem(CHAVE_TOKEN);
  localStorage.removeItem(CHAVE_USER);
  location.reload();
}

async function api(caminho, opcoes = {}) {
  const headers = { ...(opcoes.headers || {}) };
  const t = token();
  if (t) headers.Authorization = `Bearer ${t}`;
  if (opcoes.body && !(opcoes.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const resp = await fetch(caminho, { ...opcoes, headers });
  const texto = await resp.text();
  let dados = null;
  try { dados = texto ? JSON.parse(texto) : null; } catch { dados = { detail: texto }; }
  if (!resp.ok) {
    if (resp.status === 401 && t && !String(caminho).includes("/auth/login")) {
      sair();
      throw new Error("Sessão expirada.");
    }
    const d = dados && dados.detail;
    throw new Error(typeof d === "string" ? d : JSON.stringify(d || resp.status));
  }
  return dados;
}

function pct(v) { return `${(Number(v) * 100).toFixed(1).replace(".", ",")} %`; }
function nomeClasse(chave) { return NOMES_CLASSE[chave] || chave; }
function quando(iso) {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return d.toLocaleString("pt-PT");
}
function temPapel(...papeis) {
  const u = utilizador();
  return u && papeis.includes(u.funcao);
}

function mostrarApp() {
  const u = utilizador();
  $("#ecra-login").hidden = true;
  $("#ecra-app").hidden = false;
  $("#who-nome").textContent = `${u.primeiro_nome} ${u.ultimo_nome}`;
  const b = $("#who-papel");
  b.textContent = PAPEIS[u.funcao] || u.funcao;
  b.className = `badge ${u.funcao}`;
  $$("#nav-principal button[data-roles]").forEach((btn) => {
    const ok = btn.dataset.roles.split(",").includes(u.funcao);
    btn.hidden = !ok;
  });
  abrir("inicio");
}

function abrir(id) {
  $$(".seccao").forEach((s) => { s.hidden = s.id !== `sec-${id}`; });
  $$("#nav-principal button[data-sec]").forEach((b) => {
    b.classList.toggle("activo", b.dataset.sec === id);
  });
  $("#titulo-sec").textContent = TITULOS[id];
  if (id === "inicio") carregarInicio();
  if (id === "modelos") carregarModelos();
  if (id === "feedback") carregarFeedback();
  if (id === "utilizadores") carregarUsers();
}

async function carregarInicio() {
  const caixa = $("#inicio-stats");
  caixa.innerHTML = "";
  try {
    const me = await api("/auth/me");
    localStorage.setItem(CHAVE_USER, JSON.stringify(me));
    const pred = await api("/predicoes");
    const cards = [["Predições", String(pred.length)]];
    if (temPapel("radiologista", "admin")) {
      const fb = await api("/feedback");
      cards.push(["Feedback", String(fb.length)]);
    }
    if (temPapel("admin")) {
      const users = await api("/admin/utilizadores");
      cards.push(["Utilizadores", String(users.length)]);
    }
    caixa.innerHTML = cards.map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  } catch (e) {
    caixa.innerHTML = `<p class="erro">${e.message}</p>`;
  }
}

function blocoModelo(r, chave) {
  if (chave === "ensemble") return r.ensemble_oficial;
  const probs = r.probabilidades_individuais[chave];
  if (!probs) return r.ensemble_oficial;
  const [classe, confianca] = Object.entries(probs).reduce((a, b) => (b[1] > a[1] ? b : a));
  return { probabilidades: probs, classe_predita: classe, confianca };
}

function barras(probs) {
  const max = Math.max(...Object.values(probs));
  return Object.entries(probs).map(([k, v]) => `
    <div class="prob-linha ${v === max ? "vencedor" : ""}">
      <div class="row-between"><span>${nomeClasse(k)}</span><span>${pct(v)}</span></div>
      <div class="bar"><span style="width:${Math.max(0, Math.min(100, v * 100))}%"></span></div>
    </div>`).join("");
}

function htmlResultado(r, chave, verOriginal) {
  const bloco = blocoModelo(r, chave);
  const xai = r.visualizacao_xai || {};
  const mapa = xai[bloco.classe_predita] || Object.values(xai)[0] || "";
  const src = verOriginal || !mapa ? ultimaImagemUrl : mapa;
  const podeFb = temPapel("radiologista", "admin");
  const tabs = ORDEM_MODELOS.filter((k) => k === "ensemble" || r.probabilidades_individuais[k]).map((k) => `
    <button type="button" data-modelo="${k}" class="${k === chave ? "activo" : ""}">${NOMES_MODELO[k]}</button>
  `).join("");
  const pillsXai = mapa ? `
    <div class="xai-pills">
      <button type="button" data-vista="cam" class="${verOriginal ? "" : "activo"}">Grad-CAM</button>
      <button type="button" data-vista="orig" class="${verOriginal ? "activo" : ""}">Ver raio-X original</button>
    </div>` : "";

  return `
    <div class="resultado-stack">
    ${src ? `<div class="resultado-img">
      <img src="${src}" alt="Radiografia">
      ${pillsXai}
    </div>` : ""}
    <button type="button" class="btn-nova" id="btn-nova-imagem">Carregar nova imagem</button>
    <article class="card card-resultado">
      <p class="rotulo-sec">Resultado</p>
      <div class="model-tabs">${tabs}</div>
      ${r.alerta_incerteza && chave === "ensemble" ? "<p class='erro'>Alerta de incerteza (confiança &lt; 50%).</p>" : ""}
      <div class="probs">${barras(bloco.probabilidades)}</div>
      <div class="veredicto">
        <p class="classe-grande">${nomeClasse(bloco.classe_predita)}</p>
        <p class="confianca-txt">Confiança ${pct(bloco.confianca)}</p>
      </div>
      <div class="nota-clinica"><p>${r.nota_clinica}</p></div>
      ${podeFb ? `
        <div class="fb-acoes">
          <button type="button" class="btn-boa" id="btn-fb-boa">Boa resposta</button>
          <button type="button" class="btn-ruim" id="btn-fb-ruim">Resposta Ruim</button>
        </div>
        <form id="form-fb" class="fb-correcao" hidden>
          <label>Classe correcta
            <select name="classe_proposta">
              ${r.classes.map((c) => `<option value="${c}" ${c === bloco.classe_predita ? "selected" : ""}>${nomeClasse(c)}</option>`).join("")}
            </select>
          </label>
          <label>Comentário <input name="comentario" maxlength="2000" placeholder="Opcional"></label>
          <button type="submit">Submeter correcção</button>
        </form>
        <p id="fb-msg" hidden></p>` : "<p>O médico usa o CAD; o rótulo de treino só é dado por radiologista ou admin.</p>"}
    </article>
    </div>`;
}

function ligarResultado(r, chave, verOriginal) {
  $("#predict-resultado").innerHTML = htmlResultado(r, chave, verOriginal);
  $("#predict-resultado").querySelectorAll("[data-modelo]").forEach((btn) => {
    btn.addEventListener("click", () => ligarResultado(r, btn.dataset.modelo, verOriginal));
  });
  $("#predict-resultado").querySelectorAll("[data-vista]").forEach((btn) => {
    btn.addEventListener("click", () => ligarResultado(r, chave, btn.dataset.vista === "orig"));
  });
  const nova = $("#btn-nova-imagem");
  if (nova) nova.addEventListener("click", resetInferencia);
  ligarFeedback(r);
}

function ligarFeedback(r) {
  const ens = r.ensemble_oficial;
  const form = $("#form-fb");
  const msg = $("#fb-msg");
  async function enviar(classe, comentario) {
    if (!msg) return;
    try {
      const fb = await api(`/predicoes/${r.predicao_id}/feedback`, {
        method: "POST",
        body: JSON.stringify({ classe_proposta: classe, comentario: comentario || null }),
      });
      msg.hidden = false;
      msg.className = "ok-msg";
      msg.textContent = `Feedback #${fb.id} em estado ${fb.estado}. ${fb.aviso_proteccao || ""}`;
      $$(".btn-boa, .btn-ruim").forEach((b) => { b.disabled = true; });
      if (form) form.hidden = true;
    } catch (e) {
      msg.hidden = false;
      msg.className = "erro";
      msg.textContent = e.message;
    }
  }
  const boa = $("#btn-fb-boa");
  const ruim = $("#btn-fb-ruim");
  if (boa) boa.addEventListener("click", () => enviar(ens.classe_predita, "Boa resposta"));
  if (ruim) ruim.addEventListener("click", () => { if (form) form.hidden = false; });
  if (form) {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fd = new FormData(form);
      await enviar(fd.get("classe_proposta"), fd.get("comentario") || "Resposta ruim");
    });
  }
}

function pintarResultado(r) {
  ultimaResposta = r;
  const temXai = Object.keys(r.visualizacao_xai || {}).length > 0;
  ligarResultado(r, "ensemble", !temXai);
}

function estadoProcessar(activo) {
  const vazio = $("#zona-vazio");
  const proc = $("#zona-processar");
  if (!vazio || !proc) return;
  vazio.hidden = activo;
  proc.hidden = !activo;
}

function resetInferencia() {
  const form = $("#form-predict");
  const painel = $(".card-ler");
  if (painel) painel.hidden = false;
  form.reset();
  form.querySelector("[name=incluir_xai]").checked = true;
  $("#predict-resultado").innerHTML = "";
  $("#predict-erro").hidden = true;
  estadoProcessar(false);
  if (ultimaImagemUrl) URL.revokeObjectURL(ultimaImagemUrl);
  ultimaImagemUrl = "";
  ultimaResposta = null;
}

async function carregarModelos() {
  const caixa = $("#modelos-lista");
  caixa.innerHTML = "<p>A carregar fichas…</p>";
  try {
    const lista = await api("/modelos/treino");
    caixa.innerHTML = lista.map((m) => `
      <article class="card card-modelo">
        <div class="row-between">
          <h3>${m.nome}</h3>
          <span class="badge ${m.ensemble_oficial ? "aceite" : "pendente"}">${m.ensemble_oficial ? "oficial" : "comparação"}</span>
        </div>
        <p class="meta">semente ${m.seed} · ${m.entrada} · ${m.tamanho_bytes ? `${(m.tamanho_bytes / 1e6).toFixed(1)} MB` : "—"}</p>
        <p class="notas">${m.notas_estudo}</p>
        <p class="corpus">${
          Object.entries(m.corpus_feedback_aceite).map(([k, v]) => `<span>${nomeClasse(k)} ${v}</span>`).join("")
        }</p>
      </article>`).join("");
  } catch (e) {
    caixa.innerHTML = `<p class="erro">${e.message}</p>`;
  }
}

async function carregarFeedback() {
  const caixa = $("#feedback-lista");
  const corpusCard = $("#card-corpus");
  corpusCard.hidden = !temPapel("admin");
  caixa.innerHTML = "<p>A carregar…</p>";
  try {
    const lista = await api("/feedback");
    if (!lista.length) {
      caixa.innerHTML = "<p>Ainda não há pareceres.</p>";
      return;
    }
    caixa.innerHTML = lista.map((f) => `
      <div class="card" style="margin-bottom:10px">
        <div class="row-between">
          <strong>#${f.id} · predição ${f.predicao_id}</strong>
          <span class="badge ${f.estado}">${f.estado}</span>
        </div>
        <p>${f.utilizador_nome} (${NIVEIS[f.nivel_experiencia]}) · modelo ${nomeClasse(f.classe_modelo)} → proposta ${nomeClasse(f.classe_proposta)} · peso ${f.peso}</p>
        ${f.comentario ? `<p>${f.comentario}</p>` : ""}
        ${f.aviso_proteccao ? `<p><small>${f.aviso_proteccao}</small></p>` : ""}
        ${temPapel("admin") && f.estado === "pendente" ? `
          <div class="acoes">
            <button class="ok" data-rev="${f.id}" data-est="aceite">Aceitar no corpus</button>
            <button class="perigo" data-rev="${f.id}" data-est="rejeitado">Rejeitar</button>
          </div>` : ""}
        ${f.revisto_por_nome ? `<p><small>Revisto por ${f.revisto_por_nome} em ${quando(f.revisto_em)}</small></p>` : ""}
      </div>`).join("");
    caixa.querySelectorAll("[data-rev]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api(`/admin/feedback/${btn.dataset.rev}/revisao`, {
            method: "POST",
            body: JSON.stringify({ estado: btn.dataset.est }),
          });
          carregarFeedback();
        } catch (e) { alert(e.message); }
      });
    });
  } catch (e) {
    caixa.innerHTML = `<p class="erro">${e.message}</p>`;
  }
}

async function carregarCorpus() {
  const pre = $("#corpus-json");
  pre.textContent = "A exportar…";
  try {
    const dados = await api("/admin/corpus");
    pre.textContent = JSON.stringify(dados, null, 2);
  } catch (e) {
    pre.textContent = e.message;
  }
}

async function carregarUsers() {
  const tb = $("#users-tbody");
  tb.innerHTML = "<tr><td colspan='8'>A carregar…</td></tr>";
  try {
    const lista = await api("/admin/utilizadores");
    tb.innerHTML = lista.map((u) => `
      <tr>
        <td data-label="Nome">${u.primeiro_nome} ${u.ultimo_nome}</td>
        <td data-label="Email">${u.email}</td>
        <td data-label="Função">${PAPEIS[u.funcao]}</td>
        <td data-label="Nível">${NIVEIS[u.nivel_experiencia]}</td>
        <td data-label="Criada">${quando(u.criado_em)}</td>
        <td data-label="Por">${u.criado_por_nome || "sistema"}</td>
        <td data-label="Estado"><span class="badge ${u.bloqueado ? "bloqueado" : "aceite"}">${u.bloqueado ? "bloqueado" : "activo"}</span></td>
        <td class="acoes" data-label="">
          <button class="secundario" data-blk="${u.id}" data-val="${u.bloqueado ? "false" : "true"}">${u.bloqueado ? "Desbloquear" : "Bloquear"}</button>
          <button data-pw="${u.id}">Nova senha</button>
        </td>
      </tr>`).join("");
    tb.querySelectorAll("[data-blk]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api(`/admin/utilizadores/${btn.dataset.blk}`, {
            method: "PATCH",
            body: JSON.stringify({ bloqueado: btn.dataset.val === "true" }),
          });
          carregarUsers();
        } catch (e) { alert(e.message); }
      });
    });
    tb.querySelectorAll("[data-pw]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const senha = prompt("Nova senha (mín. 8 caracteres):");
        if (!senha) return;
        try {
          await api(`/admin/utilizadores/${btn.dataset.pw}/senha`, {
            method: "POST",
            body: JSON.stringify({ senha_nova: senha }),
          });
          alert("Senha actualizada. A sessão desse utilizador foi invalidada.");
        } catch (e) { alert(e.message); }
      });
    });
  } catch (e) {
    tb.innerHTML = `<tr><td colspan="8" class="erro">${e.message}</td></tr>`;
  }
}

function ligarEventos() {
  $("#form-login").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const erro = $("#login-erro");
    erro.hidden = true;
    try {
      const dados = await api("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email: fd.get("email"), senha: fd.get("senha") }),
      });
      guardarSessao(dados);
      mostrarApp();
    } catch (e) {
      erro.hidden = false;
      erro.textContent = e.message;
    }
  });

  $$("#nav-principal button[data-sec]").forEach((btn) => {
    btn.addEventListener("click", () => abrir(btn.dataset.sec));
  });
  $("#btn-sair").addEventListener("click", sair);

  const formPredict = $("#form-predict");
  formPredict.addEventListener("submit", enviarPredicao);
  formPredict.querySelector("[name=file]").addEventListener("change", () => {
    if (formPredict.querySelector("[name=file]").files[0]) formPredict.requestSubmit();
  });

  $("#form-user").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const erro = $("#user-erro");
    erro.hidden = true;
    try {
      await api("/admin/utilizadores", {
        method: "POST",
        body: JSON.stringify({
          email: fd.get("email"),
          primeiro_nome: fd.get("primeiro_nome"),
          ultimo_nome: fd.get("ultimo_nome"),
          senha: fd.get("senha"),
          funcao: fd.get("funcao"),
          nivel_experiencia: fd.get("nivel_experiencia"),
        }),
      });
      ev.target.reset();
      carregarUsers();
    } catch (e) {
      erro.hidden = false;
      erro.textContent = e.message;
    }
  });

  $("#form-senha").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const msg = $("#senha-msg");
    try {
      await api("/auth/senha", {
        method: "POST",
        body: JSON.stringify({
          senha_atual: fd.get("senha_atual"),
          senha_nova: fd.get("senha_nova"),
        }),
      });
      msg.hidden = false;
      msg.className = "ok-msg";
      msg.textContent = "Senha alterada. A iniciar sessão de novo…";
      setTimeout(sair, 800);
    } catch (e) {
      msg.hidden = false;
      msg.className = "erro";
      msg.textContent = e.message;
    }
  });

  $("#btn-corpus").addEventListener("click", carregarCorpus);
}

async function enviarPredicao(ev) {
  ev.preventDefault();
  const erro = $("#predict-erro");
  erro.hidden = true;
  const fd = new FormData(ev.target);
  const ficheiro = fd.get("file");
  if (!ficheiro || !ficheiro.size) return;
  if (ultimaImagemUrl) URL.revokeObjectURL(ultimaImagemUrl);
  ultimaImagemUrl = URL.createObjectURL(ficheiro);
  const body = new FormData();
  body.append("file", ficheiro);
  const xai = fd.get("incluir_xai") ? "true" : "false";
  estadoProcessar(true);
  $("#predict-resultado").innerHTML = "";
  try {
    const r = await api(`/predict?incluir_xai=${xai}`, { method: "POST", body });
    estadoProcessar(false);
    $("#zona-vazio").closest(".card-ler").hidden = true;
    pintarResultado(r);
  } catch (e) {
    estadoProcessar(false);
    erro.hidden = false;
    erro.textContent = e.message;
  }
}

async function arranque() {
  ligarEventos();
  if (!token()) return;
  try {
    const me = await api("/auth/me");
    localStorage.setItem(CHAVE_USER, JSON.stringify(me));
    mostrarApp();
  } catch {
    localStorage.removeItem(CHAVE_TOKEN);
    localStorage.removeItem(CHAVE_USER);
  }
}

arranque();
