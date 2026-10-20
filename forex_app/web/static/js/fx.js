// FXPulse effects — vanilla ports of Aceternity-style components.
// Everything is opt-in via data attributes / small helpers, and respects reduced-motion + touch devices.
(function () {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = matchMedia("(pointer: fine)").matches;

  /* ---- Pointer tracking: card spotlight, tilt, hero spotlight (event delegation → works for dynamic cards) ---- */
  let raf = 0, lastEv = null;
  function onMove(e) {
    lastEv = e;
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const ev = lastEv;
      const t = ev.target.closest ? ev.target : null;
      if (!t) return;
      const spot = t.closest(".fx-spot");
      if (spot) setXY(spot, ev);
      const card = t.closest(".fx-card");
      if (card) {
        setXY(card, ev);
        if (card.dataset.tilt && !reduce) tilt(card, ev, Number(card.dataset.tilt) || 6);
      }
    });
  }
  function setXY(el, ev) {
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${ev.clientX - r.left}px`);
    el.style.setProperty("--my", `${ev.clientY - r.top}px`);
  }
  function tilt(el, ev, max) {
    const r = el.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width - 0.5;
    const py = (ev.clientY - r.top) / r.height - 0.5;
    el.style.setProperty("--ry", `${(px * max).toFixed(2)}deg`);
    el.style.setProperty("--rx", `${(-py * max).toFixed(2)}deg`);
  }
  function onLeave(e) {
    const card = e.target.closest && e.target.closest(".fx-card");
    if (card && !card.contains(e.relatedTarget)) {
      card.style.setProperty("--rx", "0deg");
      card.style.setProperty("--ry", "0deg");
    }
  }
  if (fine) {
    document.addEventListener("pointermove", onMove, { passive: true });
    document.addEventListener("pointerout", onLeave, { passive: true });
  }

  /* ---- Text generate effect: words fade/blur in one after another ---- */
  function generate(el, opts = {}) {
    if (!el) return;
    const text = opts.text ?? el.dataset.text ?? el.textContent;
    el.dataset.text = text;
    el.setAttribute("aria-label", text);
    el.innerHTML = text.split(/\s+/).filter(Boolean).map((w, i) =>
      `<span class="fx-word" aria-hidden="true" style="--i:${i};--d:${opts.delay || 0}ms">${escape(w)}</span>`).join(" ");
  }

  /* ---- Flip words ---- */
  function flipWords(el, words, every = 2600) {
    if (!el || !words.length) return;
    let i = 0;
    el.classList.add("fx-flip");
    const show = (w) => { el.innerHTML = `<span class="in"><span class="g">${escape(w)}</span></span>`; };
    show(words[0]);
    if (reduce) return;
    setInterval(() => {
      if (document.hidden) return;
      const cur = el.firstElementChild;
      if (cur) cur.className = "out";
      setTimeout(() => { i = (i + 1) % words.length; show(words[i]); }, 340);
    }, every);
  }

  /* ---- Animated number ---- */
  function countUp(el, to, ms = 1100) {
    if (!el) return;
    const from = Number((el.dataset.n || "0").replace(/,/g, "")) || 0;
    el.dataset.n = String(to);
    if (reduce || from === to) { el.textContent = Number(to).toLocaleString(); return; }
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / ms);
      const e = 1 - Math.pow(1 - p, 4);
      el.textContent = Math.round(from + (to - from) * e).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /* ---- Meteors ---- */
  function meteors(container, n = 10) {
    if (!container || reduce) return;
    const wrap = document.createElement("div");
    wrap.className = "fx-meteors";
    wrap.innerHTML = Array.from({ length: n }, () =>
      `<i style="--l:${Math.round(Math.random() * 110)}%;--dl:${(Math.random() * 8).toFixed(1)}s;--t:${(3 + Math.random() * 5).toFixed(1)}s"></i>`).join("");
    container.prepend(wrap);
  }

  /* ---- Stars + shooting stars canvas ---- */
  function stars(canvas, { count = 140 } = {}) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let w = 0, h = 0, dpr = Math.min(devicePixelRatio || 1, 2), S = [], shooters = [];
    function size() {
      w = canvas.clientWidth; h = canvas.clientHeight;
      canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      S = Array.from({ length: Math.round(count * Math.min(1, (w * h) / 1000000 + 0.4)) }, () => ({
        x: Math.random() * w, y: Math.random() * h, r: Math.random() * 1.2 + 0.2,
        a: Math.random(), s: 0.004 + Math.random() * 0.012, d: Math.random() > 0.5 ? 1 : -1,
      }));
    }
    function draw() {
      ctx.clearRect(0, 0, w, h);
      for (const s of S) {
        s.a += s.s * s.d;
        if (s.a > 1 || s.a < 0.1) s.d *= -1;
        ctx.globalAlpha = Math.max(0.1, Math.min(1, s.a)) * 0.85;
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 6.283); ctx.fill();
      }
      for (let i = shooters.length - 1; i >= 0; i--) {
        const m = shooters[i];
        m.x += m.vx; m.y += m.vy; m.life -= 1;
        const g = ctx.createLinearGradient(m.x, m.y, m.x - m.vx * 14, m.y - m.vy * 14);
        g.addColorStop(0, `rgba(${m.c},1)`); g.addColorStop(1, `rgba(${m.c},0)`);
        ctx.globalAlpha = Math.min(1, m.life / 20);
        ctx.strokeStyle = g; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(m.x, m.y); ctx.lineTo(m.x - m.vx * 14, m.y - m.vy * 14); ctx.stroke();
        if (m.life <= 0 || m.x > w + 50 || m.y > h + 50) shooters.splice(i, 1);
      }
      ctx.globalAlpha = 1;
    }
    function shoot() {
      const c = ["165,180,252", "103,232,249", "134,239,172"][Math.floor(Math.random() * 3)];
      const sp = 9 + Math.random() * 6;
      shooters.push({ x: Math.random() * w * 0.7, y: Math.random() * h * 0.4, vx: sp, vy: sp * 0.55, life: 70, c });
    }
    size();
    addEventListener("resize", size);
    if (reduce) { draw(); return; }
    let run = true, next = 0;
    document.addEventListener("visibilitychange", () => { run = !document.hidden; if (run) loop(); });
    function loop(t = 0) {
      if (!run) return;
      draw();
      if (t > next) { shoot(); next = t + 2500 + Math.random() * 4000; }
      requestAnimationFrame(loop);
    }
    loop();
  }

  /* ---- Floating navbar: hides while scrolling down, returns on scroll up ---- */
  function floatingNav(el, scroller = window) {
    if (!el) return;
    let last = 0;
    const get = () => (scroller === window ? window.scrollY : scroller.scrollTop);
    scroller.addEventListener("scroll", () => {
      const y = get();
      if (Math.abs(y - last) < 6) return;
      el.classList.toggle("hidden-nav", y > last && y > 140);
      last = y;
    }, { passive: true });
  }

  /* ---- Sparkle burst (e.g. on like) ---- */
  function burst(x, y, colors = ["#fb7185", "#f472b6", "#fbbf24", "#a5b4fc"], n = 14) {
    if (reduce) return;
    for (let i = 0; i < n; i++) {
      const el = document.createElement("i");
      const a = (i / n) * Math.PI * 2 + Math.random() * 0.4;
      const d = 40 + Math.random() * 50;
      el.className = "fx-spark";
      el.style.cssText = `left:${x}px;top:${y}px;--c:${colors[i % colors.length]};--dx:${Math.cos(a) * d}px;--dy:${Math.sin(a) * d}px`;
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 850);
    }
  }

  function escape(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  window.FX = { generate, flipWords, countUp, meteors, stars, floatingNav, burst, reduce };
})();
