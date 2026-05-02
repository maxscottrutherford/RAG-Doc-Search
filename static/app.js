const uploadForm = document.getElementById("upload-form");
const uploadFile = document.getElementById("upload-file");
const uploadStatus = document.getElementById("upload-status");

const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const searchStatus = document.getElementById("search-status");
const searchResults = document.getElementById("search-results");

function setStatus(el, type, message) {
  el.className = "status" + (type ? ` ${type}` : "");
  el.textContent = message || "";
}

function formatScore(score) {
  if (score == null || Number.isNaN(Number(score))) return "—";
  return Number(score).toFixed(4);
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = uploadFile.files[0];
  if (!file) {
    setStatus(uploadStatus, "error", "Please choose a file.");
    return;
  }

  setStatus(uploadStatus, "info", "Uploading…");
  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch("/api/v1/documents/upload", {
      method: "POST",
      body: fd,
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = data.detail;
      const msg =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || d).join("; ")
            : res.statusText;
      setStatus(uploadStatus, "error", msg || "Upload failed.");
      return;
    }

    setStatus(
      uploadStatus,
      "success",
      `Indexed “${data.filename}”: ${data.chunks_stored} chunk(s) stored.`
    );
    uploadForm.reset();
  } catch (err) {
    setStatus(uploadStatus, "error", err.message || "Network error.");
  }
});

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    setStatus(searchStatus, "error", "Enter a question.");
    return;
  }

  setStatus(searchStatus, "info", "Searching…");
  searchResults.classList.add("hidden");
  searchResults.innerHTML = "";

  try {
    const res = await fetch("/api/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = data.detail;
      const msg =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || d).join("; ")
            : res.statusText;
      setStatus(searchStatus, "error", msg || "Search failed.");
      return;
    }

    setStatus(searchStatus, "success", "Done.");
    renderSearchResults(data);
  } catch (err) {
    setStatus(searchStatus, "error", err.message || "Network error.");
  }
});

function renderSearchResults(data) {
  const answer = escapeHtml(data.answer || "");
  const chunks = data.source_chunks || [];
  const ev = data.evaluation;

  let evalHtml = "";
  if (ev) {
    evalHtml = `
      <div class="eval-block">
        <strong>Evaluation</strong> (LLM judge):
        relevance ${ev.relevance}/5 · completeness ${ev.completeness}/5 ·
        groundedness ${ev.groundedness}/5
        ${ev.notes ? ` — ${escapeHtml(ev.notes)}` : ""}
      </div>`;
  }

  const chunksHtml = chunks
    .map(
      (ch, i) => `
      <details class="chunk-details">
        <summary>
          <span>Source ${i + 1}: ${escapeHtml(ch.document_id || "—")}</span>
          <span class="chunk-meta">chunk id ${escapeHtml(String(ch.chunk_id))}</span>
          <span class="score-badge">similarity ${formatScore(ch.score)}</span>
        </summary>
        <pre class="chunk-body">${escapeHtml(ch.text || "")}</pre>
      </details>`
    )
    .join("");

  searchResults.innerHTML = `
    <div class="answer-card">
      <h3>Answer</h3>
      <p class="answer-body">${answer}</p>
      ${evalHtml}
    </div>
    <h3 class="chunks-heading">Source chunks</h3>
    ${chunks.length ? chunksHtml : "<p class=\"hint\">No chunks returned.</p>"}
  `;

  searchResults.classList.remove("hidden");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
