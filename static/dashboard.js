const CHAVE_TOKEN = "cxr_jwt";
const CHAVE_USER = "cxr_user";
const TITULOS = {
  inicio: ["Início", "Protocolo de feedback e papéis"],
  inferencia: ["Testar imagem", "Inferência com o ensemble oficial"],
  modelos: ["Dados de treino", "Ficha de cada CNN (semente 42)"],
  feedback: ["Feedback", "Pareceres pendentes e corpus revisto"],
  utilizadores: ["Utilizadores", "Registar, bloquear e actualizar senhas"],
  conta: ["A minha senha", "A alteração invalida a sessão actual"],
};
const PAPEIS = { medico: "Médico", radiologista: "Radiologista", admin: "Administrador" };
const NIVEIS = { junior: "Júnior", pleno: "Pleno", senior: "Sénior" };

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

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

function pct(v) { return `${(Number(v) * 100).toFixed(1)}%`; }
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
  const [t, s] = TITULOS[id];
  $("#titulo-sec").textContent = t;
  $("#sub-sec").textContent = s;
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
    const cards = [
      ["Conta", `${me.primeiro_nome} ${me.ultimo_nome}`],
      ["Função", PAPEIS[me.funcao]],
      ["Nível", NIVEIS[me.nivel_experiencia] || me.nivel_experiencia],
      ["Predições", String(pred.length)],
      ["Criada por", me.criado_por_nome || "sistema"],
    ];
    if (temPapel("radiologista", "admin")) {
      const fb = await api("/feedback");
      cards.push(["Feedbacks", String(fb.length)]);
    }
    caixa.innerHTML = cards.map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  } catch (e) {
    caixa.innerHTML = `<p class="erro">${e.message}</p>`;
  }
}

function barras(probs) {
  return Object.entries(probs).map(([k, v]) => `
    <div>
      <div class="row-between"><span>${k}</span><span class="mono">${pct(v)}</span></div>
      <div class="bar"><span style="width:${Math.max(0, Math.min(100, v * 100))}%"></span></div>
    </div>`).join("");
}

function pintarResultado(r) {
  const ens = r.ensemble_oficial;
  const xai = Object.entries(r.visualizacao_xai || {}).map(([k, src]) =>
    `<figure><img src="${src}" alt="${k}"><figcaption>${k}</figcaption></figure>`
  ).join("");
  const podeFb = temPapel("radiologista", "admin");
  $("#predict-resultado").innerHTML = `
    <article class="card">
      <p class="eyebrow">predição #${r.predicao_id}</p>
      <h3>Ensemble oficial: ${ens.classe_predita} · ${pct(ens.confianca)}</h3>
      ${r.alerta_incerteza ? "<p class='erro'>Alerta de incerteza (confiança &lt; 50%).</p>" : ""}
      <div class="probs">${barras(ens.probabilidades)}</div>
      <p>${r.nota_clinica}</p>
      <p><small>${r.protocolo_feedback}</small></p>
      ${xai ? `<div class="xai">${xai}</div>` : ""}
      ${podeFb ? `
        <form id="form-fb" class="form-grid">
          <label>Classe correcta (parecer)
            <select name="classe_proposta">
              ${r.classes.map((c) => `<option ${c === ens.classe_predita ? "selected" : ""}>${c}</option>`).join("")}
            </select>
          </label>
          <label>Comentário <input name="comentario" maxlength="2000" placeholder="Opcional"></label>
          <button type="submit">Submeter feedback</button>
        </form>
        <p id="fb-msg" hidden></p>` : "<p>O médico usa o CAD; o rótulo de treino só é dado por radiologista ou admin.</p>"}
    </article>`;
  const form = $("#form-fb");
  if (form) {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fd = new FormData(form);
      const msg = $("#fb-msg");
      try {
        const fb = await api(`/predicoes/${r.predicao_id}/feedback`, {
          method: "POST",
          body: JSON.stringify({
            classe_proposta: fd.get("classe_proposta"),
            comentario: fd.get("comentario") || null,
          }),
        });
        msg.hidden = false;
        msg.className = "ok-msg";
        msg.textContent = `Feedback #${fb.id} em estado ${fb.estado}. ${fb.aviso_proteccao || ""}`;
        form.querySelector("button").disabled = true;
      } catch (e) {
        msg.hidden = false;
        msg.className = "erro";
        msg.textContent = e.message;
      }
    });
  }
}

async function carregarModelos() {
  const caixa = $("#modelos-lista");
  caixa.innerHTML = "<p>A carregar fichas…</p>";
  try {
    const lista = await api("/modelos/treino");
    caixa.innerHTML = lista.map((m) => `
      <article class="card">
        <p class="eyebrow">${m.ensemble_oficial ? "ensemble oficial" : "comparação"}</p>
        <h3>${m.nome}</h3>
        <p class="mono">${m.ficheiro}</p>
        <p>Semente ${m.seed} · ${m.entrada} · ${m.tamanho_bytes ? `${(m.tamanho_bytes / 1e6).toFixed(1)} MB` : "—"}</p>
        <p>${m.notas_estudo}</p>
        <p><strong>Corpus aceite (global):</strong> ${
          Object.entries(m.corpus_feedback_aceite).map(([k, v]) => `${k} ${v}`).join(" · ")
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
        <p>${f.utilizador_nome} (${NIVEIS[f.nivel_experiencia]}) · modelo ${f.classe_modelo} → proposta ${f.classe_proposta} · peso ${f.peso}</p>
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
        <td>${u.primeiro_nome} ${u.ultimo_nome}</td>
        <td>${u.email}</td>
        <td>${PAPEIS[u.funcao]}</td>
        <td>${NIVEIS[u.nivel_experiencia]}</td>
        <td>${quando(u.criado_em)}</td>
        <td>${u.criado_por_nome || "sistema"}</td>
        <td><span class="badge ${u.bloqueado ? "bloqueado" : "aceite"}">${u.bloqueado ? "bloqueado" : "activo"}</span></td>
        <td class="acoes">
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

  $("#form-predict").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const erro = $("#predict-erro");
    erro.hidden = true;
    const fd = new FormData(ev.target);
    const body = new FormData();
    body.append("file", fd.get("file"));
    const xai = fd.get("incluir_xai") ? "true" : "false";
    $("#predict-resultado").innerHTML = "<p>A inferir…</p>";
    try {
      const r = await api(`/predict?incluir_xai=${xai}`, { method: "POST", body });
      pintarResultado(r);
    } catch (e) {
      $("#predict-resultado").innerHTML = "";
      erro.hidden = false;
      erro.textContent = e.message;
    }
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
