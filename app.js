// 表示したい Markdown ファイルのリスト
// path はリポジトリ内の相対パス
const pages = [
  { id: "intro", title: "LangeveinMonteCarlo", path: "docs/1-LangeveinMonteCarlo.md" },
  { id: "guide", title: "ガイド", path: "docs/guide.md" },
  { id: "api", title: "API リファレンス", path: "docs/api.md" }
];

// クエリパラメータ ?page=intro で現在のページを保持
function getCurrentPageId() {
  const params = new URLSearchParams(window.location.search);
  const p = params.get("page");
  if (p && pages.some(page => page.id === p)) {
    return p;
  }
  return pages[0].id; // デフォルト
}

function setCurrentPageId(id) {
  const params = new URLSearchParams(window.location.search);
  params.set("page", id);
  const newUrl =
    window.location.pathname + "?" + params.toString() + window.location.hash;
  history.pushState({ page: id }, "", newUrl);
}

function renderSidebar(currentId) {
  const listEl = document.getElementById("sidebar-list");
  listEl.innerHTML = "";

  pages.forEach(page => {
    const a = document.createElement("button");
    a.textContent = page.title;
    a.className = "sidebar-item" + (page.id === currentId ? " active" : "");
    a.type = "button";
    a.addEventListener("click", () => {
      if (page.id !== getCurrentPageId()) {
        setCurrentPageId(page.id);
        loadPage(page.id);
      }
    });
    listEl.appendChild(a);
  });
}

async function loadPage(id) {
  const page = pages.find(p => p.id === id);
  if (!page) return;

  const contentEl = document.getElementById("content");
  contentEl.textContent = "読み込み中...";

  try {
    const res = await fetch(page.path);
    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }
    const mdText = await res.text();

    // marked で HTML に変換
    const html = marked.parse(mdText, {
      mangle: false,
      headerIds: true
    });
    contentEl.innerHTML = html;

    // コードブロックのハイライト
    document
      .querySelectorAll("#content pre code")
      .forEach(block => hljs.highlightElement(block));

    // サイドバーの選択状態を更新
    renderSidebar(id);
  } catch (err) {
    console.error(err);
    contentEl.textContent = "読み込みに失敗しました。";
  }
}

// 初期化
window.addEventListener("DOMContentLoaded", () => {
  const currentId = getCurrentPageId();
  renderSidebar(currentId);
  loadPage(currentId);
});

// ブラウザ戻る／進む対応
window.addEventListener("popstate", event => {
  const id = getCurrentPageId();
  renderSidebar(id);
  loadPage(id);
});
