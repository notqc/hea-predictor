// Browser/Node port of hea_core feature pipeline + saved SVR/GPR models (exported from the Python models).
function makeEngine(M) {
  const E = M.elements, R = 8.314462618;
  function fractions(comp) {
    const tot = E.reduce((a, e) => a + (comp[e] || 0), 0);
    const f = {}; E.forEach(e => { if ((comp[e] || 0) > 0) f[e] = comp[e] / tot; }); return f;
  }
  function descriptors(f) {
    const els = Object.keys(f), c = els.map(e => f[e]), el = els.map(e => M.el[e]);
    const idx = els.map(e => E.indexOf(e));
    let dH = 0; for (let i = 0; i < els.length; i++) for (let j = i + 1; j < els.length; j++) dH += 4 * M.H[idx[i]][idx[j]] * c[i] * c[j];
    const dS = -R * c.reduce((a, x) => a + x * Math.log(x), 0);
    const sum = (fn) => c.reduce((a, x, i) => a + x * fn(i), 0);
    const rbar = sum(i => el[i].r), delta = 100 * Math.sqrt(sum(i => (1 - el[i].r / rbar) ** 2));
    const vec = sum(i => el[i].VEC), chib = sum(i => el[i].chi), dchi = Math.sqrt(sum(i => (el[i].chi - chib) ** 2));
    const tm = sum(i => el[i].Tm), omega = Math.min(tm * dS / Math.max(Math.abs(dH) * 1000, 1e-6), 1000);
    const g = sum(i => el[i].G), dG = 100 * Math.sqrt(sum(i => (1 - el[i].G / g) ** 2)), ec = sum(i => el[i].Ec);
    return { dH_mix: dH, dS_mix: dS, delta, VEC: vec, d_chi: dchi, Tm: tm, Omega: omega, log_Omega: Math.log10(omega),
      G_mean: g, delta_G: dG, E_coh: ec, labusch_param: g * (delta / 100) ** (4 / 3) };
  }
  function features(f, d) { return M.features.map(k => k.startsWith('x_') ? (f[k.slice(2)] || 0) : d[k]); }
  function svr(x) {
    const S = M.svr, z = x.map((v, i) => (v - S.mu[i]) / S.sd[i]);
    let y = S.b;
    for (let k = 0; k < S.sv.length; k++) { let d2 = 0; const v = S.sv[k]; for (let i = 0; i < z.length; i++) d2 += (z[i] - v[i]) ** 2; y += S.dc[k] * Math.exp(-S.gamma * d2); }
    return y;
  }
  function gpr(x) {
    const G = M.gpr, z = x.map((v, i) => (v - G.mu[i]) / G.sd[i]), n = G.X.length;
    const ks = new Array(n);
    for (let k = 0; k < n; k++) { let d2 = 0; const v = G.X[k]; for (let i = 0; i < z.length; i++) d2 += ((z[i] - v[i]) / G.ls[i]) ** 2;
      const d = Math.sqrt(d2), s5 = Math.sqrt(5) * d; ks[k] = G.C * (1 + s5 + 5 * d2 / 3) * Math.exp(-s5); }
    let mean = 0; for (let k = 0; k < n; k++) mean += ks[k] * G.alpha[k];
    const v = new Array(n); let vv = 0;
    for (let i = 0; i < n; i++) { const Li = G.L[i]; let s = ks[i]; for (let j = 0; j < i; j++) s -= Li[j] * v[j]; v[i] = s / Li[i]; vv += v[i] * v[i]; }
    const varr = Math.max(G.C + G.noise - vv, 0);
    return { mean: mean * G.ys + G.ym, sd: Math.sqrt(varr) * G.ys };
  }
  function domain(f, d, x) {
    const D = M.dom, w = [];
    E.forEach(e => { const v = f[e] || 0; if (v > D.emax[e] + 1e-9) w.push(`${e} = ${(100 * v).toFixed(1)} at.% is above the training maximum (${(100 * D.emax[e]).toFixed(1)} at.%).`); });
    Object.keys(D.dmin).forEach(k => { if (d[k] < D.dmin[k] - 1e-9 || d[k] > D.dmax[k] + 1e-9) w.push(`Descriptor ${k} = ${d[k].toPrecision(3)} is outside the training range ${D.dmin[k].toPrecision(3)} – ${D.dmax[k].toPrecision(3)}.`); });
    const z = x.map((v, i) => (v - D.mu[i]) / D.sd[i]); let nn = Infinity;
    for (const t of D.T) { let d2 = 0; for (let i = 0; i < z.length; i++) d2 += (z[i] - t[i]) ** 2; if (d2 < nn) nn = d2; }
    nn = Math.sqrt(nn);
    if (nn > D.thr) w.push('The alloy is far from every training alloy (nearest-neighbour distance above the 95th percentile of training distances); treat the prediction as an extrapolation.');
    return { inDomain: w.length === 0, warnings: w, nn };
  }
  function predict(comp) {
    const f = fractions(comp), d = descriptors(f), x = features(f, d);
    return { f, d, HV: svr(x), gp: gpr(x), dom: domain(f, d, x) };
  }
  return { predict, descriptors, fractions };
}
if (typeof module !== 'undefined') module.exports = makeEngine;
