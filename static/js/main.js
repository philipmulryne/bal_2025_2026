// Theme toggle
(function(){
  const root = document.documentElement;
  const saved = localStorage.getItem("theme");
  if (saved) { root.setAttribute("data-theme", saved); }
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.addEventListener("click", ()=>{
      const cur = root.getAttribute("data-theme") || "dark";
      const nxt = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", nxt);
      localStorage.setItem("theme", nxt);
    });
  }
})();

// Search filter & shortcuts on index
(function(){
  const input = document.getElementById("search");
  const cards = Array.from(document.querySelectorAll(".card"));
  if (!input || cards.length === 0) return;

  // Focus with '/'
  window.addEventListener("keydown", (e)=>{
    if (e.key === "/" && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    }
    // numeric shortcuts 1..9
    const num = parseInt(e.key, 10);
    if (!isNaN(num) && num >= 1 && num <= 9) {
      const target = cards.find(c => c.querySelector(".shortcut")?.textContent.trim() === String(num));
      if (target) { window.location.href = target.getAttribute("href"); }
    }
  });

  // Filter cards by name + desc
  input.addEventListener("input", ()=>{
    const q = input.value.trim().toLowerCase();
    cards.forEach(card=>{
      const hay = card.getAttribute("data-name") || "";
      card.style.display = hay.includes(q) ? "" : "none";
    });
  });
})();
