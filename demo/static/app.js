(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const secs = (v) => (v == null ? "–" : v >= 60 ? `${Math.floor(v / 60)} m ${Math.round(v % 60)} s` : `${Number(v).toFixed(1)} s`);

  const state = { cfg: null, leads: [], room: null, roomName: null, call: null, poll: null, captions: new Map(), busy: false };

  // ---------- small helpers ----------
  async function api(path, body) {
    const res = await fetch(path, body === undefined ? { headers: { Accept: "application/json" } }
      : { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
    return data;
  }
  const showError = (msg) => { const e = $("#error"); e.hidden = !msg; e.textContent = msg || ""; };
  const setStatus = (text, tone) => { const s = $("#status"); s.textContent = text; s.className = `pill ${tone || ""}`; };
  const selectedLead = () => state.leads.find((l) => l.id === $("#lead").value);

  // ---------- lead ----------
  const LEAD_ROWS = [["Name", "name"], ["Phone", "phone"], ["Project", "project_interest"], ["Status", "status"], ["Budget", "budget"],
    ["Area", "preferred_area"], ["Property", "property_type"], ["Timeline", "timeline"], ["Interest", "interest_level"], ["Callback", "callback_at"]];
  function leadInfo(lead) {
    if (!lead) return "";
    return LEAD_ROWS.map(([label, key]) => `<dt>${esc(label)}</dt><dd>${esc(lead[key] == null || lead[key] === "" ? "–" : key === "budget" ? `${Number(lead[key]).toLocaleString()} USD` : lead[key])}</dd>`).join("")
      + `<dt>Notes</dt><dd>${esc((lead.notes || []).length)}</dd>`;
  }
  function renderLeads(keep) {
    const sel = $("#lead");
    sel.innerHTML = state.leads.map((l) => `<option value="${esc(l.id)}">${esc(l.id)} · ${esc(l.name)} · ${esc(l.status)}</option>`).join("");
    if (keep) sel.value = keep;
    $("#leadInfo").innerHTML = leadInfo(selectedLead());
  }
  async function loadLeads(keep) { state.leads = await api("/api/leads"); renderLeads(keep); }

  // ---------- prompt ----------
  let checkTimer = null;
  function promptChanged() {
    const text = $("#prompt").value;
    $("#promptCount").textContent = `${text.length} / ${state.cfg.max_prompt_chars}`;
    clearTimeout(checkTimer);
    checkTimer = setTimeout(checkPrompt, 350);
  }
  async function checkPrompt() {
    const msg = $("#promptMsg");
    try {
      const r = await api("/api/prompt/check", { prompt: $("#prompt").value });
      msg.className = r.ok ? "muted" : "muted bad";
      msg.textContent = r.ok ? (r.warning || (isDefaultPrompt() ? "Default prompt" : "Custom prompt: OK")) : r.error;
      state.promptOk = r.ok;
    } catch (e) { msg.className = "muted bad"; msg.textContent = e.message; state.promptOk = false; }
    updateStart();
  }
  const isDefaultPrompt = () => $("#prompt").value.trim() === state.cfg.default_prompt.trim();

  // ---------- setup ----------
  async function init() {
    try {
      state.cfg = await api("/api/config");
      await loadLeads();
    } catch (e) { showError(e.message); setStatus("Server error", "crit"); return; }
    const cfg = state.cfg;
    $("#prompt").value = cfg.default_prompt;
    $("#promptHelp").innerHTML = `<b>Placeholders:</b> ${cfg.placeholders.map((p) => `<code>{${esc(p)}}</code>`).join(" ")}<br><br>` +
      `<b>Tools the agent uses:</b> ${cfg.tool_names.map((p) => `<code>${esc(p)}</code>`).join(" ")}<br><br>To write a real { or } type it twice.`;
    const group = (g, name) => `<optgroup label="${name}">${cfg.voices.filter((v) => v.gender === g).map((v) => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join("")}</optgroup>`;
    $("#voice").innerHTML = group("female", "Female") + group("male", "Male");
    $("#voice").value = cfg.default_voice;
    $("#stt").innerHTML = cfg.stt.map((o) => `<option value="${esc(o.id)}" ${o.available || o.id === "default" ? "" : "disabled"}>${esc(o.label)}${o.available || o.id === "default" ? "" : " (not running)"}</option>`).join("");
    $("#models").textContent = `Text-to-speech engine: ${cfg.models.tts} (one engine, the voice is selectable). LLM: ${cfg.models.llm}.`;
    $("#prompt").addEventListener("input", promptChanged);
    $("#promptReset").addEventListener("click", () => { $("#prompt").value = cfg.default_prompt; promptChanged(); });
    $("#lead").addEventListener("change", () => { $("#leadInfo").innerHTML = leadInfo(selectedLead()); });
    $("#resetLead").addEventListener("click", resetLead);
    $("#start").addEventListener("click", startCall);
    $("#end").addEventListener("click", () => endCall("You ended the call."));
    state.promptOk = true;
    promptChanged();
    setStatus("Ready", "good");
  }
  function updateStart() { $("#start").disabled = state.busy || !state.promptOk; }
  function lockSetup(locked) {
    state.busy = locked;
    ["#lead", "#resetLead", "#prompt", "#promptReset", "#voice", "#stt"].forEach((s) => { $(s).disabled = locked; });
    $("#start").hidden = locked; $("#end").hidden = !locked;
    updateStart();
  }
  async function resetLead() {
    try { const id = $("#lead").value; await api(`/api/leads/${encodeURIComponent(id)}/reset`, {}); await loadLeads(id); showError(""); }
    catch (e) { showError(e.message); }
  }

  // ---------- the call ----------
  function bubble(role, text, key) {
    let el = key ? state.captions.get(key) : null;
    if (!el) {
      el = document.createElement("div");
      el.className = `line ${role}`;
      $("#captions").appendChild(el);
      if (key) state.captions.set(key, el);
    }
    el.textContent = text;
    el.scrollIntoView({ block: "nearest" });
  }
  async function startCall() {
    showError(""); $("#result").innerHTML = ""; $("#captions").innerHTML = ""; state.captions.clear();
    clearInterval(state.poll);
    const LK = window.LivekitClient;
    if (!LK) { showError("The LiveKit browser library did not load (check the internet connection)."); return; }
    lockSetup(true);
    $("#live").textContent = "Starting the call…"; setStatus("Connecting…", "warn");
    let call;
    try {
      call = await api("/api/call", { lead_id: $("#lead").value, prompt: $("#prompt").value, tts_voice: $("#voice").value, stt_id: $("#stt").value });
    } catch (e) { showError(e.message); lockSetup(false); setStatus("Ready", "good"); $("#live").textContent = "The call did not start."; return; }
    state.call = call; state.roomName = call.room;
    const room = new LK.Room({ audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true } });
    state.room = room;
    room.on(LK.RoomEvent.TrackSubscribed, (track) => { if (track.kind === LK.Track.Kind.Audio) $("#audio").appendChild(track.attach()); });
    room.on(LK.RoomEvent.ParticipantConnected, () => { setStatus("Live", "good"); $("#live").textContent = "The agent joined and will greet you. Speak after the greeting."; });
    room.on(LK.RoomEvent.ParticipantDisconnected, () => endCall("The agent ended the call."));
    room.on(LK.RoomEvent.Disconnected, () => endCall("The call is over."));
    room.registerTextStreamHandler("lk.transcription", async (reader, info) => {
      const text = (await reader.readAll()).trim();
      if (!text) return;
      const seg = (reader.info.attributes || {})["lk.segment_id"];
      bubble(info.identity === room.localParticipant.identity ? "lead" : "agent", text, seg);
    });
    try {
      await room.connect(call.url, call.token);
      await room.localParticipant.setMicrophoneEnabled(true);
      await room.startAudio();
      setStatus("Waiting for the agent…", "warn");
      $("#live").textContent = "Connected. Waiting for the agent to pick up (the first call is slowest while the models load)…";
    } catch (e) {
      showError(`Could not connect: ${e.message}. If the microphone was blocked, allow it in the browser and try again.`);
      endCall("Could not connect.", true);
    }
  }
  let ending = false;
  async function endCall(reason, failed) {
    if (ending || !state.roomName) return;
    ending = true;
    const name = state.roomName;
    try { if (state.room) await state.room.disconnect(); } catch (e) { /* already closed */ }
    state.room = null;
    $("#audio").innerHTML = "";
    lockSetup(false);
    api(`/api/call/${encodeURIComponent(name)}/end`, {}).catch(() => {});
    ending = false;
    state.roomName = null;
    if (failed) { setStatus("Ready", "good"); $("#live").textContent = reason; return; }
    setStatus("Saving the call…", "warn");
    $("#live").textContent = `${reason} Saving the call record and running the post-call check (slow on CPU)…`;
    pollResult(name);
  }
  function pollResult(name) {
    const started = Date.now();
    clearInterval(state.poll);
    const tick = async () => {
      try {
        const r = await api(`/api/call/${encodeURIComponent(name)}/result`);
        if (r.status === "done") {
          clearInterval(state.poll); setStatus("Ready", "good"); $("#live").textContent = "Call finished.";
          await loadLeads($("#lead").value).catch(() => {});
          renderResult(r);
        } else if (Date.now() - started > 10 * 60 * 1000) {
          clearInterval(state.poll); setStatus("Ready", "good");
          $("#live").textContent = "The call record has not appeared after 10 minutes. Check that the agent worker is running (DEMO_MODE=1).";
        } else {
          $("#live").textContent = r.status === "processing" ? "The post-call check is running…" : "Waiting for the call record…";
        }
      } catch (e) { /* server busy: try again */ }
    };
    state.poll = setInterval(tick, 2500); tick();
  }

  // ---------- result ----------
  const prettyJson = (v) => {
    if (v == null || v === "") return "";
    let o = v;
    if (typeof v === "string") { try { o = JSON.parse(v); } catch (e) { return v; } }
    return typeof o === "string" ? o : JSON.stringify(o, null, 2);
  };
  const toolCard = (t) => {
    const ok = !t.is_error;
    return `<div class="toolcard ${ok ? "" : "bad"}"><div class="tc-head"><b>${esc(t.name)}</b><span class="muted">${esc((t.at || "").slice(11, 19))}</span><span class="pill ${ok ? "good" : "crit"}">${ok ? "Worked" : "Failed"}</span></div>`
      + `<div class="tc-label">Sent to the tool</div><div class="code">${esc(prettyJson(t.arguments) || "(nothing)")}</div>`
      + `<div class="tc-label">${ok ? "Result" : "Error"}</div><div class="code">${esc(prettyJson(t.output) || "(nothing returned)")}</div></div>`;
  };
  function renderResult(r) {
    const m = r.models || {}, v = r.versions || {}, s = (r.settings || {}).demo || {};
    const box = (k, val) => `<div><span>${esc(k)}</span><b>${esc(val ?? "–")}</b></div>`;
    const tx = (r.transcript || []).map((t) => `<div class="line ${t.role === "agent" ? "agent" : "lead"}">${esc(t.text)}</div>`).join("") || '<div class="muted">No conversation.</div>';
    const tools = (r.tool_calls || []).length ? `<div class="steps">${r.tool_calls.map(toolCard).join("")}</div>` : '<div class="muted">The agent did not use any tool.</div>';
    const crm = (r.crm_changes || []).map((c) => {
      const rows = Object.entries(c.changed || {}).map(([k, x]) => `<tr><td>${esc(k)}</td><td>${esc(prettyJson(x.before) || "–")}</td><td class="arrow">→</td><td><b>${esc(prettyJson(x.after) || "–")}</b></td></tr>`).join("");
      return `<div class="rblock"><div class="muted">${esc(c.action || "change")} · ${c.source === "post_call" ? "post-call check" : "during the call"} · ${c.ok ? "saved" : "FAILED"}</div>`
        + (rows ? `<table><thead><tr><th>Field</th><th>Before</th><th></th><th>After</th></tr></thead><tbody>${rows}</tbody></table>` : "") + (c.error ? `<div class="note bad">${esc(c.error)}</div>` : "") + "</div>";
    }).join("") || '<div class="muted">No CRM changes were made.</div>';
    const errors = (r.errors || []).length ? `<div class="note bad">${r.errors.map((e) => `${esc(e.stage)}: ${esc(e.error || "")}`).join("<br>")}</div>` : "";
    $("#result").innerHTML = `<div class="sum">${box("Result", r.classification)}${box("Length", secs(r.duration_seconds))}${box("Ended by", r.ended_by)}${box("Lead status now", (r.lead_now || {}).status)}</div>`
      + (r.summary ? `<div class="rblock"><h3>Summary</h3><div class="note">${esc(r.summary)}</div></div>` : "") + errors
      + `<div class="rblock"><h3>Conversation</h3><div class="tx">${tx}</div></div>`
      + `<div class="rblock"><h3>Tools: what the agent sent and what came back</h3>${tools}</div>`
      + `<div class="rblock"><h3>CRM changes (before → after)</h3>${crm}</div>`
      + `<div class="rblock"><h3>What was used</h3><div class="sum">${box("LLM", m.llm)}${box("Speech-to-text", m.stt)}${box("Text-to-speech", m.tts)}${box("Voice", m.tts_voice)}${box("Prompt version", v.prompt)}${box("Custom prompt", s.custom_prompt ? "Yes" : "No")}</div></div>`;
  }

  init();
})();
