const minSup = document.getElementById("minSup");
const minConf = document.getElementById("minConf");
const minSupVal = document.getElementById("minSupVal");
const minConfVal = document.getElementById("minConfVal");
minSup.addEventListener("input", () => {
  minSupVal.textContent = parseFloat(minSup.value).toFixed(3);
});
minConf.addEventListener("input", () => {
  minConfVal.textContent = parseFloat(minConf.value).toFixed(2);
});

const trainStatus = document.getElementById("trainStatus");
const predictStatus = document.getElementById("predictStatus");
const btnPredict = document.getElementById("btnPredict");
const results = document.getElementById("results");
const csvFileInput = document.getElementById("csvFile");
const csvFileName = document.getElementById("csvFileName");

csvFileInput?.addEventListener("change", () => {
  const selectedFile = csvFileInput.files?.[0];
  csvFileName.textContent = selectedFile
    ? selectedFile.name
    : "Chưa chọn file nào";
});

document.getElementById("btnTrain").addEventListener("click", async () => {
  trainStatus.style.display = "block";
  trainStatus.className = "status info";
  trainStatus.textContent = "Đang xử lý…";
  try {
    const selectedFile = csvFileInput?.files?.[0];
    let r;
    if (selectedFile) {
      const formData = new FormData();
      formData.append("csv_file", selectedFile);
      formData.append("min_support", parseFloat(minSup.value).toString());
      formData.append("min_confidence", parseFloat(minConf.value).toString());
      r = await fetch("/api/train", {
        method: "POST",
        body: formData,
      });
    } else {
      r = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv_path: document.getElementById("csvPath").value,
          min_support: parseFloat(minSup.value),
          min_confidence: parseFloat(minConf.value),
        }),
      });
    }
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || "Lỗi huấn luyện");
    trainStatus.className = "status ok";
    trainStatus.textContent =
      `Đã nạp ${data.n_transactions} giao dịch · ${data.n_itemsets} tập phổ biến · ` +
      `luật: chẩn đoán ${data.n_rules_dx}, mức độ ${data.n_rules_sev}, điều trị ${data.n_rules_tx}.`;
    btnPredict.disabled = false;
  } catch (e) {
    trainStatus.className = "status err";
    trainStatus.textContent = e.message || String(e);
    btnPredict.disabled = true;
  }
});

function selectedSymptoms() {
  return Array.from(
    document.querySelectorAll('input[name="symptom"]:checked'),
  ).map((x) => x.value);
}

function renderList(container, items, emptyMsg) {
  if (!items || !items.length) {
    container.innerHTML = `<p class="empty">${emptyMsg}</p>`;
    return;
  }
  container.innerHTML = items
    .map((it) => {
      const pct = Math.min(100, Math.round(it.confidence * 100));
      return `<div class="item">
              <div><strong>${escapeHtml(it.label)}</strong>
                <div class="bar"><i style="width:${pct}%"></i></div>
              </div>
              <div class="metrics">confidence ${it.confidence.toFixed(3)}<br/>lift ${it.lift.toFixed(3)}</div>
            </div>`;
    })
    .join("");
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

document.getElementById("btnPredict").addEventListener("click", async () => {
  predictStatus.style.display = "none";
  const syms = selectedSymptoms();
  if (!syms.length) {
    predictStatus.style.display = "block";
    predictStatus.textContent = "Hãy chọn ít nhất một triệu chứng.";
    return;
  }
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms: syms }),
    });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || "Lỗi gợi ý");
    results.style.display = "block";
    renderList(
      document.getElementById("outDx"),
      data.diagnoses,
      "Không có luật chẩn đoán khớp — thử giảm ngưỡng hoặc đổi tập triệu chứng.",
    );
    renderList(
      document.getElementById("outSev"),
      data.severities,
      "Không có luật gợi ý mức độ từ tập triệu chứng / chẩn đoán hàng đầu.",
    );
    renderList(
      document.getElementById("outTx"),
      data.treatments,
      "Không có luật gợi ý điều trị từ tập triệu chứng / chẩn đoán hàng đầu.",
    );
  } catch (e) {
    predictStatus.style.display = "block";
    predictStatus.textContent = e.message || String(e);
  }
});

document.getElementById("btnClear").addEventListener("click", () => {
  document
    .querySelectorAll('input[name="symptom"]')
    .forEach((x) => (x.checked = false));
});
