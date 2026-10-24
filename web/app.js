// Общий код экранов. Опрос сервера раз в 2 с; звук — локальный файл /alarm.wav.
(function () {
  const role = document.body.dataset.role;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtTs = (s) => s ? new Date(s).toLocaleString("ru-RU", { timeZone: "UTC", hour12: false }) + " UTC" : "—";
  const fmtNum = (v) => v === null || v === undefined ? "—" : Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4);
  const ago = (s, now) => { if (!s || !now) return ""; const m = Math.round((new Date(now) - new Date(s)) / 60000);
    return m < 60 ? `${m} мин назад` : m < 2880 ? `${Math.round(m / 60)} ч назад` : `${Math.round(m / 1440)} сут назад`; };
  // Пометки М2, которые стоит показать оператору (служебные вроде state_derived скрыты).
  const FLAG_TITLES = { outlier: "выброс", nonphysical: "нефизичное", non_numeric: "нечисловое", gap: "пропуск",
    restored: "восстановлено", downtime: "простой", zero_flow_pressure: "давление при нулевом расходе", excluded: "исключено" };
  const shownFlags = (f) => (f || "").split(";").filter((x) => FLAG_TITLES[x]).map((x) => FLAG_TITLES[x]).join(", ");
  let reasons = {}, soundOn = false, playedFor = null, ackFor = null;
  const audio = $("#alarm-sound");

  const soundBtn = $("#sound-toggle");
  soundBtn.addEventListener("click", () => {
    soundOn = !soundOn;
    soundBtn.textContent = soundOn ? "Звук включён" : "Включить звук";
    soundBtn.classList.toggle("primary", soundOn);
    if (soundOn) { audio.play().then(() => audio.pause()).catch(() => {}); audio.currentTime = 0; }
  });
  const ackBtn = $("#sound-ack");
  if (ackBtn) ackBtn.addEventListener("click", () => { ackFor = playedFor; audio.pause(); audio.loop = false; });

  function reasonOptions() {
    return Object.entries(reasons).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join("");
  }
  function decideForm(e) {
    if (e.decision) return `<span class="badge ${e.decision === "confirm" ? "ok" : "muted"}">${e.decision === "confirm" ? "подтверждено" : "отклонено"}</span>
      <span class="muted">${esc(reasons[e.reason_code] || e.reason_code)} · ${esc(e.decided_by)}</span>`;
    return `<form class="decide" data-id="${e.id}">
      <select name="reason_code" required><option value="">причина…</option>${reasonOptions()}</select>
      <input name="user_name" placeholder="ФИО" required value="${esc(localStorage.getItem("ft-user") || "")}">
      <input name="comment" placeholder="комментарий">
      <button type="submit" name="decision" value="confirm" class="primary">Подтвердить</button>
      <button type="submit" name="decision" value="reject">Отклонить</button></form>`;
  }
  document.addEventListener("submit", async (ev) => {
    const f = ev.target.closest("form.decide"); if (!f) return;
    ev.preventDefault();
    const fd = new FormData(f);
    const body = { event_id: +f.dataset.id, decision: ev.submitter.value, reason_code: fd.get("reason_code"),
      user_name: fd.get("user_name"), role, comment: fd.get("comment") || null };
    try { localStorage.setItem("ft-user", body.user_name); } catch (e) {}
    const r = await fetch("/api/decision", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { alert((await r.json()).detail || "Ошибка"); return; }
    if (playedFor === body.event_id) { audio.pause(); audio.loop = false; }
    refresh();
  });

  function srcBadge(e) {
    return e.source === "asu" ? `<span class="badge src-asu">АСУ ТП</span>` : `<span class="badge src-twin">Двойник</span>`;
  }
  function eventRows(list, withForm) {
    if (!list.length) return `<tr><td colspan="4" class="empty">нет</td></tr>`;
    return list.map((e) => `<tr><td class="num">${fmtTs(e.ts)}</td><td>${srcBadge(e)} <b>${esc(e.code)}</b>
      ${e.shadow ? '<span class="badge shadow">теневой режим</span>' : ""}</td>
      <td>${esc(e.message)}${withForm && e.source !== "asu" ? decideForm(e) : ""}</td></tr>`).join("");
  }

  function renderAlert(st) {
    const box = $("#alert"); if (!box) return;
    const e = st.events.find((x) => x.id === st.sound_event_id);
    if (!e) { box.classList.remove("on"); box.innerHTML = ""; return; }
    box.classList.add("on");
    const delay = e.delay_s != null ? ` · от первого срабатывания модели ${Math.round(e.delay_s / 60)} мин` : "";
    box.innerHTML = `<h2>Предупреждение: гидраты в абсорбере ${esc(e.object_id)}</h2>
      <p>${esc(fmtTs(e.ts))}${delay}</p><p><b>Действия:</b> ${esc(e.actions)}</p>${decideForm(e)}`;
    if (soundOn && playedFor !== e.id && ackFor !== e.id) { audio.loop = true; audio.currentTime = 0; audio.play().catch(() => {}); playedFor = e.id; }
  }

  function renderSnapshot(snap) {
    $("#now").textContent = snap ? "Время данных: " + fmtTs(snap.now) : "Нет данных: цикл двойника ещё не запускался";
    if (!snap) return;
    $("#groups").innerHTML = (snap.groups[role] || []).map((g) => `<h3>${esc(g.title)}</h3><table>
      <tr><th>Объект</th><th>Параметр</th><th class="num">Значение</th><th>Время</th></tr>
      ${g.rows.map((r) => `<tr><td>${esc(r.object_id)}</td><td>${esc(r.title)}</td>
        <td class="num">${fmtNum(r.value)} <span class="muted">${esc(r.unit)}</span>
        ${shownFlags(r.flags) ? `<div class="flags">${esc(shownFlags(r.flags))}</div>` : ""}</td>
        <td class="muted">${ago(r.ts, snap.now)}</td></tr>`).join("")}</table>`).join("");
    $("#freshness").innerHTML = `<tr><th>Поток</th><th>Состояние</th><th>Последнее значение</th></tr>` +
      snap.freshness.map((f) => `<tr><td>${esc(f.title)}</td>
        <td>${f.stale ? '<span class="badge stale">устарел</span>' : '<span class="badge ok">свежий</span>'}</td>
        <td class="muted">${fmtTs(f.last_ts)} (${ago(f.last_ts, snap.now)})</td></tr>`).join("");
    const g = snap.geotech;
    $("#geotech").innerHTML = `<tr><th>Пункт</th><th>Состояние</th><th class="num">Осадка, мм/год</th><th>Последний радар</th></tr>` +
      g.objects.map((o) => `<tr><td>${esc(o.object_id)}</td><td><span class="badge ${o.color}">${esc(o.title)}</span></td>
        <td class="num">${o.rate_mm_per_year == null ? "—" : o.rate_mm_per_year.toFixed(1)}</td>
        <td class="muted">${o.last_radar ? fmtTs(o.last_radar) : "нет (зимой радара нет)"}</td></tr>`).join("");
    const lm = snap.last_measurements;
    $("#last").innerHTML = `<tr><th>Что</th><th>Последний замер</th></tr>` +
      `<tr><td>Спутниковый радар (основания)</td><td>${fmtTs(lm.geotech_radar)}</td></tr>` +
      Object.entries(lm.ecp).map(([k, v]) => `<tr><td>ЭХЗ ${esc(k)} (ручной замер)</td><td>${fmtTs(v)} <span class="muted">${ago(v, snap.now)}</span></td></tr>`).join("") +
      Object.entries(lm.pipelines_survey).map(([k, v]) => `<tr><td>Обследование газопровода ${esc(k)}</td><td>${fmtTs(v)}</td></tr>`).join("");
    const hy = $("#hydrates");
    if (hy) hy.innerHTML = snap.hydrates ? snap.hydrates.map((h) => `<tr><td>${esc(h.object_id)}</td>
      <td class="num">${(h.proba * 100).toFixed(0)} %</td><td class="num">${h.positives_20min}</td></tr>`).join("")
      : `<tr><td colspan="3" class="empty">модель гидратов не обучена</td></tr>`;
  }

  async function refresh() {
    try {
      const r = await fetch(`/api/state?role=${role}`); const st = await r.json();
      reasons = st.reasons;
      renderSnapshot(st.snapshot);
      renderAlert(st);
      const twin = st.events.filter((e) => !(e.id === st.sound_event_id));
      $("#events").innerHTML = eventRows(twin, true);
      const asu = $("#asu"); if (asu) asu.innerHTML = eventRows(st.asu_alarms, false);
      $("#conn").textContent = "";
    } catch (e) { $("#conn").textContent = "нет связи с сервером"; }
  }
  refresh();
  setInterval(refresh, 2000);
})();
