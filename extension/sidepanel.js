const $ = id => document.getElementById(id);

function add(text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  $("log").appendChild(d);
  $("log").scrollTop = $("log").scrollHeight;
  return d;
}

async function refreshStatus() {
  chrome.runtime.sendMessage({ type: "agentStatus" }, (r) => {
    const ok = r && r.connected;
    $("status").textContent = ok ? "● на связи" : "○ нет связи";
    $("status").className = ok ? "ok" : "off";
    $("setup").style.display = ok ? "none" : "block";
  });
}

$("save").addEventListener("click", async () => {
  await chrome.storage.local.set({ agentToken: $("token").value.trim() });
  chrome.runtime.reload();
});

async function send() {
  const text = $("inp").value.trim();
  if (!text) return;
  $("inp").value = "";
  add(text, "me");
  const thinking = add("…", "ag");
  chrome.runtime.sendMessage({ type: "agentChat", text }, (r) => {
    thinking.textContent = r && r.ok ? r.result : "⚠ " + ((r && r.error) || "нет ответа");
    refreshStatus();
  });
}

$("send").addEventListener("click", send);
$("inp").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
refreshStatus();
setInterval(refreshStatus, 4000);
