const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("file-input");
const runBtn = $("run-btn");
const frame = $("frame");
const preview = $("preview");
const placeholder = $("placeholder");
const badge = $("badge");
const bgIcons = document.querySelector(".bg-icons");

let selectedFile = null;

function setBadge(text, state) {
  badge.textContent = text;
  badge.className = "badge " + state;
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  resetResults();
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  placeholder.hidden = true;
  frame.classList.add("loaded");
  runBtn.disabled = false;
  setBadge("PHOTO LOADED", "idle");
  dropzone.querySelector(".dz-title").textContent = file.name;
}

fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

document.addEventListener("mousemove", (e) => {
  const mx = (e.clientX / window.innerWidth - 0.5) * 2;
  const my = (e.clientY / window.innerHeight - 0.5) * 2;
  bgIcons.style.setProperty("--mx", mx.toFixed(3));
  bgIcons.style.setProperty("--my", my.toFixed(3));
});

const scanline = $("scanline");
const logEl = $("log");
let lastDetections = [];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

async function typeLog(text) {
  logEl.textContent = "";
  for (const ch of text) {
    logEl.textContent += ch;
    await sleep(18);
  }
}

function setChip(id, text, state) {
  const chip = $(id);
  chip.textContent = text;
  chip.className = "chip " + state;
}

function resetResults() {
  $("boxes").replaceChildren();
  lastDetections = [];
  $("detections").replaceChildren(el("li", "empty", "Nothing yet"));
  $("trace").replaceChildren(el("li", "empty", "Run an inspection to see the agent's steps"));
  setChip("chip-s3", "S3 saved", "");
  setChip("chip-sns", "Alert sent", "");
  logEl.textContent = "";
  scanline.hidden = true;
  frame.style.removeProperty("--orbit-color");
}

function imageRect() {
  const fw = frame.clientWidth, fh = frame.clientHeight;
  const iw = preview.naturalWidth, ih = preview.naturalHeight;
  const scale = Math.min(fw / iw, fh / ih);
  return { x: (fw - iw * scale) / 2, y: (fh - ih * scale) / 2, scale };
}

const MIN_CONF = 0.5;

function kind(d) {
  const bad = d.violation ?? d.potential_violation;
  if (!bad) return "ok";
  return d.conf >= MIN_CONF ? "violation" : "uncertain";
}

function renderBoxes(list) {
  const layer = $("boxes");
  layer.replaceChildren();
  const r = imageRect();
  list.filter((d) => d.conf >= MIN_CONF).forEach((d, i) => {
    const k = kind(d);
    const [x1, y1, x2, y2] = d.box;
    const b = el("div", "box" + (k === "violation" ? "" : " ok"));
    b.style.left = r.x + x1 * r.scale + "px";
    b.style.top = r.y + y1 * r.scale + "px";
    b.style.width = (x2 - x1) * r.scale + "px";
    b.style.height = (y2 - y1) * r.scale + "px";
    b.style.animationDelay = i * 0.25 + "s";
    b.append(el("span", "", `${d.label} ${Math.round(d.conf * 100)}%`));
    layer.appendChild(b);
  });
}

function renderDetections(list) {
  const ul = $("detections");
  ul.replaceChildren();
  if (!list.length) {
    ul.append(el("li", "empty", "No people detected"));
    return;
  }
  const order = { violation: 0, uncertain: 1, ok: 2 };
  const sorted = [...list].sort((a, b) => order[kind(a)] - order[kind(b)] || b.conf - a.conf);
  sorted.forEach((d, i) => {
    const k = kind(d);
    const pct = Math.round(d.conf * 100);
    const li = el("li", k);
    const fill = el("i");
    const bar = el("div", "bar");
    bar.append(fill);
        const flagText = {
      violation: "VIOLATION",
      uncertain: "UNCERTAIN",
      ok: d.label.startsWith("No-") ? "NOT REQUIRED" : "OK",
    }[k];
    li.append(el("span", "label", d.label), el("span", "conf", pct + "%"),
              el("span", "flag", flagText), bar);
    ul.append(li);
    setTimeout(() => (fill.style.width = pct + "%"), 150 + i * 150);
  });
}

function summarise(t) {
  if (t.check === "quality") {
    const q = t.result;
    return ["Image quality", q.acceptable
      ? `OK (sharpness ${q.blur_score.toFixed(0)}, brightness ${q.brightness.toFixed(0)})`
      : "Poor: blurry, too dark or too bright"];
  }
  if (t.check === "initial_detection") return ["Detection", `${t.result.detection_count} objects found`];
  if (t.check === "targeted_crop_comparison") return ["Zoom re-check", `${t.result.uncertain_count} uncertain detections re-checked`];
  if (t.check === "enhanced_full_image_comparison") {
    if (!t.result) return ["Enhanced re-check", "skipped (nothing uncertain)"];
    return ["Enhanced re-check", t.result.error ? "could not run" : "compared with a contrast-enhanced copy"];
  }
  if (t.action) return ["Decision: " + t.action, t.reason || t.status || ""];
  return ["Step", JSON.stringify(t)];
}

function renderTrace(trace) {
  const ol = $("trace");
  ol.replaceChildren();
  trace.forEach((t, i) => {
    const [title, detail] = summarise(t);
    const li = el("li");
    li.append(el("span", "step-no", "STEP " + (t.step ?? i)), el("span", "", title),
              el("span", "step-detail", detail));
    ol.append(li);
    setTimeout(() => li.classList.add("show"), 300 + i * 350);
  });
}

const ACTIONS = {
  alert: { text: "ALERT: VIOLATION", cls: "alert", color: "#ff3b3b" },
  pass: { text: "PASS: ALL CLEAR", cls: "pass", color: "#2ecc8f" },
  human_approval: { text: "NEEDS HUMAN REVIEW", cls: "human", color: "#ff9f1c" },
};

const STATUSES = {
  POTENTIAL_PPE_ISSUES: { text: "POTENTIAL PPE ISSUE", cls: "alert", color: "#ff3b3b" },
  UNCERTAIN_FINDINGS: { text: "UNCERTAIN: REVIEW", cls: "human", color: "#ff9f1c" },
  IMAGE_QUALITY_INSUFFICIENT: { text: "IMAGE TOO POOR", cls: "human", color: "#ff9f1c" },
  NO_HIGH_CONFIDENCE_VIOLATION_LABELS_DETECTED: { text: "NO VIOLATIONS FOUND", cls: "pass", color: "#2ecc8f" },
};

function showResult(data) {
  const a = STATUSES[data.status] || ACTIONS[data.action] || ACTIONS.human_approval;
  setBadge(a.text, a.cls);
  frame.style.setProperty("--orbit-color", a.color);
  typeLog(data.summary ? data.summary.message : `decision: ${data.action}`);

  lastDetections = data.detections;
  renderBoxes(data.detections);
  renderDetections(data.detections);
  renderTrace(data.trace);

  setChip("chip-s3", data.saved_to_s3 ? "Saved to S3" : "Not saved (S3 off)", data.saved_to_s3 ? "on" : "off");
    if (data.status === "POTENTIAL_PPE_ISSUES") {
    setChip("chip-sns", data.alert_sent ? "Alert email sent" : "Alert not sent (SNS off)", data.alert_sent ? "on" : "off");
  } else {
    setChip("chip-sns", "No alert needed", "");
  }
}

async function runInspection() {
  if (!selectedFile) return;
  resetResults();
  runBtn.disabled = true;
  scanline.hidden = false;
  setBadge("ANALYSING…", "working");

  const form = new FormData();
  form.append("file", selectedFile);
  const request = fetch("/inspect", { method: "POST", body: form }).then(async (r) => {
    if (!r.ok) throw new Error((await r.json()).detail || "Request failed");
    return r.json();
  });
  request.catch(() => {});

  try {
    for (const step of ["checking image quality…", "detecting helmets and vests…", "deciding what to do…"]) {
      await typeLog(step);
      await sleep(450);
    }
    const data = await request;
    scanline.hidden = true;
    showResult(data);
  } catch (err) {
    scanline.hidden = true;
    setBadge("ERROR", "alert");
    typeLog(String(err.message || err));
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runInspection);
window.addEventListener("resize", () => lastDetections.length && renderBoxes(lastDetections));