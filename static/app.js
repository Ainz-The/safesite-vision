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

function renderBoxes(list) {
  const layer = $("boxes");
  layer.replaceChildren();
  const r = imageRect();
  list.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.box;
    const b = el("div", "box" + (d.violation ? "" : " ok"));
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
  list.forEach((d, i) => {
    const pct = Math.round(d.conf * 100);
    const li = el("li", d.violation ? "violation" : "ok");
    const fill = el("i");
    const bar = el("div", "bar");
    bar.append(fill);
    li.append(el("span", "label", d.label), el("span", "conf", pct + "%"),
              el("span", "flag", d.violation ? "VIOLATION" : "OK"), bar);
    ul.append(li);
    setTimeout(() => (fill.style.width = pct + "%"), 150 + i * 150);
  });
}

function renderTrace(trace) {
  const ol = $("trace");
  ol.replaceChildren();
  trace.forEach((t, i) => {
    const parts = Object.entries(t)
      .filter(([k]) => k !== "step")
      .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
    const li = el("li");
    li.append(el("span", "step-no", "STEP " + (t.step ?? i)), el("span", "", parts[0] || ""),
              el("span", "step-detail", parts.slice(1).join("  ·  ")));
    ol.append(li);
    setTimeout(() => li.classList.add("show"), 300 + i * 350);
  });
}

const ACTIONS = {
  alert: { text: "ALERT: VIOLATION", cls: "alert", color: "#ff3b3b" },
  pass: { text: "PASS: ALL CLEAR", cls: "pass", color: "#2ecc8f" },
  human_approval: { text: "NEEDS HUMAN REVIEW", cls: "human", color: "#ff9f1c" },
};

function showResult(data) {
  const a = ACTIONS[data.action] || ACTIONS.human_approval;
  setBadge(a.text, a.cls);
  frame.style.setProperty("--orbit-color", a.color);
  typeLog(`decision: ${data.action}   run ${data.run_id}`);

  lastDetections = data.detections;
  renderBoxes(data.detections);
  renderDetections(data.detections);
  renderTrace(data.trace);

  setChip("chip-s3", data.saved_to_s3 ? "Saved to S3" : "Not saved (S3 off)", data.saved_to_s3 ? "on" : "off");
  if (data.action === "alert") {
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