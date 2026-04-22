import { useState, useCallback, useMemo } from "react";
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, Area, AreaChart, Cell
} from "recharts";

// ─── THEME ────────────────────────────────────────────────────────────────────
function makeTheme(dark) {
  return dark ? {
    bg: "#0f172a", surface: "#111827", card: "#1f2937", border: "#334155",
    accent1: "#38bdf8", accent2: "#f87171", accent3: "#f59e0b",
    accent4: "#34d399", accent5: "#a78bfa",
    text: "#f1f5f9", muted: "#94a3b8", win: "#34d399", lose: "#f87171", neutral: "#cbd5e1",
  } : {
    bg: "#f8fafc", surface: "#ffffff", card: "#ffffff", border: "#cbd5e1",
    accent1: "#0369a1", accent2: "#b91c1c", accent3: "#b45309",
    accent4: "#047857", accent5: "#6d28d9",
    text: "#0f172a", muted: "#475569", win: "#047857", lose: "#b91c1c", neutral: "#334155",
  };
}
let C = makeTheme(true);

// ─── MATH UTILITIES ───────────────────────────────────────────────────────────
function normalCDF(x) {
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return 0.5 * (1.0 + sign * y);
}

function tCDF(t, df) {
  // approximation using normal for large df, beta for small
  if (df > 30) return normalCDF(t);
  const x = df / (df + t * t);
  // regularized incomplete beta approximation
  let sum = 0, term = 1;
  for (let i = 1; i <= 100; i++) {
    term *= x * (i - 0.5) / i;
    sum += term;
  }
  const p = 0.5 * (1 + (t >= 0 ? 1 : -1) * (1 - Math.sqrt(x) * (1 + sum)));
  return Math.max(0, Math.min(1, p));
}

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sampleMean(mu, sigma, n) {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += mu + sigma * randn();
  return sum / n;
}

function tTest(xA, xB, sigmaA, sigmaB, nA, nB) {
  const se = Math.sqrt(sigmaA * sigmaA / nA + sigmaB * sigmaB / nB);
  if (se === 0) return { t: 0, p: 1 };
  const t = (xB - xA) / se;
  const df = nA + nB - 2;
  const p = 2 * (1 - tCDF(Math.abs(t), df));
  return { t, p: Math.min(1, Math.max(0, p)) };
}

function bayesianUpdate(priorMean, priorStrength, priorVar, data, sigma, n) {
  const posteriorStrength = priorStrength + n;
  const posteriorMean = (priorStrength * priorMean + n * data) / posteriorStrength;
  const posteriorVar = sigma * sigma / posteriorStrength;
  return { mean: posteriorMean, variance: posteriorVar, strength: posteriorStrength };
}

function probBGreaterA(postA, postB, nSamples = 2000) {
  let count = 0;
  const sdA = Math.sqrt(postA.variance);
  const sdB = Math.sqrt(postB.variance);
  for (let i = 0; i < nSamples; i++) {
    const sA = postA.mean + sdA * randn();
    const sB = postB.mean + sdB * randn();
    if (sB > sA) count++;
  }
  return count / nSamples;
}

function computeRequiredN(alpha, power, sigma, delta) {
  if (Math.abs(delta) < 1e-10) return Infinity;
  const za = 1.96, zb = 0.842; // alpha=0.05, power=0.80
  return Math.ceil(2 * Math.pow((za + zb) * sigma / delta, 2));
}

function linspace(start, end, n) {
  const arr = [];
  for (let i = 0; i < n; i++) arr.push(start + (end - start) * i / (n - 1));
  return arr;
}

// ─── MAIN SIMULATION ─────────────────────────────────────────────────────────
function runSimulation(params, nTrials = 150) {
  const {
    samplesPerDayMin, samplesPerDayMax,
    durationMin, durationMax,
    sigmaMin, sigmaMax,
    effectMin, effectMax,
    baselineMean,
    priorMean, priorStrength, priorVar,
  } = params;

  const N_POINTS = 10;
  const spdRange = linspace(samplesPerDayMin, samplesPerDayMax, N_POINTS);
  const durRange = linspace(durationMin, durationMax, N_POINTS);
  const sigRange = linspace(sigmaMin, sigmaMax, N_POINTS);
  const effRange = linspace(effectMin / 100, effectMax / 100, N_POINTS);
  const baseRange = linspace(baselineMean * 0.5, baselineMean * 1.5, N_POINTS);
  const priorMeanRange = linspace(priorMean - 0.5 * Math.abs(priorMean + 0.01), priorMean + 0.5 * Math.abs(priorMean + 0.01), N_POINTS);
  const priorStrRange = linspace(1, Math.max(priorStrength * 2, 100), N_POINTS);

  // ── Power vs sample size ──
  const powerVsSampleSize = spdRange.map(spd => {
    const totalN = Math.round(spd * ((durationMin + durationMax) / 2));
    const n = Math.max(10, Math.round(totalN / 2));
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = baselineMean * (effectMin + effectMax) / 200;
    const muA = baselineMean, muB = baselineMean + effect;
    let freqPower = 0, bayesPower = 0;
    for (let t = 0; t < nTrials; t++) {
      const xA = sampleMean(muA, sigma, n);
      const xB = sampleMean(muB, sigma, n);
      const { p } = tTest(xA, xB, sigma, sigma, n, n);
      if (p < 0.05) freqPower++;
      const postA = bayesianUpdate(priorMean, priorStrength, priorVar, xA, sigma, n);
      const postB = bayesianUpdate(priorMean + effect, priorStrength, priorVar, xB, sigma, n);
      const pb = probBGreaterA(postA, postB, 500);
      if (pb > 0.95) bayesPower++;
    }
    return {
      sampleSize: totalN,
      freqPower: freqPower / nTrials,
      bayesPower: bayesPower / nTrials,
      required: computeRequiredN(0.05, 0.8, sigma, effect),
    };
  });

  // ── P-value distribution ──
  const pValueDist = (() => {
    const totalN = Math.round(((samplesPerDayMin + samplesPerDayMax) / 2) * ((durationMin + durationMax) / 2));
    const n = Math.max(10, Math.round(totalN / 2));
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = baselineMean * (effectMin + effectMax) / 200;
    const muA = baselineMean, muB = baselineMean + effect;
    const buckets = Array(20).fill(0);
    const buckets_null = Array(20).fill(0);
    for (let t = 0; t < nTrials * 2; t++) {
      const xA = sampleMean(muA, sigma, n);
      const xB = sampleMean(muB, sigma, n);
      const { p } = tTest(xA, xB, sigma, sigma, n, n);
      const idx = Math.min(19, Math.floor(p * 20));
      buckets[idx]++;
      // null hypothesis
      const xA2 = sampleMean(muA, sigma, n);
      const xB2 = sampleMean(muA, sigma, n); // same mu
      const { p: p2 } = tTest(xA2, xB2, sigma, sigma, n, n);
      const idx2 = Math.min(19, Math.floor(p2 * 20));
      buckets_null[idx2]++;
    }
    return buckets.map((v, i) => ({
      bucket: `${(i * 5).toFixed(0)}-${((i + 1) * 5).toFixed(0)}%`,
      withEffect: v,
      nullHypothesis: buckets_null[i],
    }));
  })();

  // ── P(B>A) trajectory over days ──
  const pbaTraj = (() => {
    const spd = Math.round((samplesPerDayMin + samplesPerDayMax) / 2);
    const maxDays = durationMax;
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = baselineMean * (effectMin + effectMax) / 200;
    const muA = baselineMean, muB = baselineMean + effect;
    const dayPointCount = Math.min(20, Math.max(2, Math.round(maxDays)));
    const days = Array.from(
      new Set(linspace(1, maxDays, dayPointCount).map(d => Math.max(1, Math.round(d))))
    ).sort((a, b) => a - b);
    return days.map(d => {
      const n = Math.max(5, spd * d);
      let pbaSum = 0, pbaMin = 1, pbaMax = 0;
      const M = 30;
      for (let t = 0; t < M; t++) {
        const xA = sampleMean(muA, sigma, n);
        const xB = sampleMean(muB, sigma, n);
        const postA = bayesianUpdate(priorMean, priorStrength, priorVar, xA, sigma, n);
        const postB = bayesianUpdate(priorMean + effect, priorStrength, priorVar, xB, sigma, n);
        const pba = probBGreaterA(postA, postB, 300);
        pbaSum += pba;
        pbaMin = Math.min(pbaMin, pba);
        pbaMax = Math.max(pbaMax, pba);
      }
      const mean = pbaSum / M;
      return { day: d, pba: mean, low: pbaMin, high: pbaMax, band: Math.max(0, pbaMax - pbaMin) };
    });
  })();

  // ── Effect size estimation error ──
  const effectEstimation = effRange.map(eff => {
    const totalN = Math.round(((samplesPerDayMin + samplesPerDayMax) / 2) * ((durationMin + durationMax) / 2));
    const n = Math.max(10, Math.round(totalN / 2));
    const sigma = (sigmaMin + sigmaMax) / 2;
    const trueEffect = baselineMean * eff;
    const muA = baselineMean, muB = baselineMean + trueEffect;
    let estimates = [];
    for (let t = 0; t < nTrials; t++) {
      const xA = sampleMean(muA, sigma, n);
      const xB = sampleMean(muB, sigma, n);
      estimates.push(xB - xA);
    }
    estimates.sort((a, b) => a - b);
    const mean = estimates.reduce((s, v) => s + v, 0) / estimates.length;
    const p25 = estimates[Math.floor(estimates.length * 0.25)];
    const p75 = estimates[Math.floor(estimates.length * 0.75)];
    const p5 = estimates[Math.floor(estimates.length * 0.05)];
    const p95 = estimates[Math.floor(estimates.length * 0.95)];
    return {
      trueEffect: (eff * 100).toFixed(1) + "%",
      trueVal: trueEffect,
      estimatedMean: mean,
      p25, p75, p5, p95,
      bias: mean - trueEffect,
    };
  });

  // ── Decision concordance heatmap ──
  const concordance = (() => {
    const rows = [];
    const nLevels = [100, 500, 1000, 5000];
    const effLevels = [0.01, 0.05, 0.10, 0.20];
    const sigma = (sigmaMin + sigmaMax) / 2;
    for (const n of nLevels) {
      for (const eff of effLevels) {
        const effect = baselineMean * eff;
        const muA = baselineMean, muB = baselineMean + effect;
        let agree = 0;
        const M = 80;
        for (let t = 0; t < M; t++) {
          const xA = sampleMean(muA, sigma, n);
          const xB = sampleMean(muB, sigma, n);
          const { p } = tTest(xA, xB, sigma, sigma, n, n);
          const freqDec = p < 0.05 ? "B" : "none";
          const postA = bayesianUpdate(priorMean, priorStrength, priorVar, xA, sigma, n);
          const postB = bayesianUpdate(priorMean + effect, priorStrength, priorVar, xB, sigma, n);
          const pba = probBGreaterA(postA, postB, 300);
          const bayesDec = pba > 0.95 ? "B" : pba < 0.05 ? "A" : "none";
          if (freqDec === bayesDec) agree++;
        }
        rows.push({
          n: n.toString(),
          effect: (eff * 100).toFixed(0) + "%",
          concordance: agree / M,
        });
      }
    }
    return rows;
  })();

  // ── FDR vs sample size ──
  const fdrVsN = spdRange.map(spd => {
    const totalN = Math.round(spd * ((durationMin + durationMax) / 2));
    const n = Math.max(10, Math.round(totalN / 2));
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = baselineMean * (effectMin + effectMax) / 200;
    const muA = baselineMean;
    let fp = 0, tp = 0, bayesFP = 0, bayesTP = 0;
    for (let t = 0; t < nTrials; t++) {
      // null case
      const xA0 = sampleMean(muA, sigma, n);
      const xB0 = sampleMean(muA, sigma, n);
      const { p: p0 } = tTest(xA0, xB0, sigma, sigma, n, n);
      if (p0 < 0.05) fp++;
      const postA0 = bayesianUpdate(priorMean, priorStrength, priorVar, xA0, sigma, n);
      const postB0 = bayesianUpdate(priorMean, priorStrength, priorVar, xB0, sigma, n);
      const pba0 = probBGreaterA(postA0, postB0, 300);
      if (pba0 > 0.95) bayesFP++;
      // effect case
      const xA1 = sampleMean(muA, sigma, n);
      const xB1 = sampleMean(muA + effect, sigma, n);
      const { p: p1 } = tTest(xA1, xB1, sigma, sigma, n, n);
      if (p1 < 0.05) tp++;
      const postA1 = bayesianUpdate(priorMean, priorStrength, priorVar, xA1, sigma, n);
      const postB1 = bayesianUpdate(priorMean + effect, priorStrength, priorVar, xB1, sigma, n);
      const pba1 = probBGreaterA(postA1, postB1, 300);
      if (pba1 > 0.95) bayesTP++;
    }
    const freqFDR = fp / (fp + tp + 0.001);
    const bayesFDR = bayesFP / (bayesFP + bayesTP + 0.001);
    return { sampleSize: totalN, freqFDR, bayesFDR };
  });

  // ── Baseline sensitivity ──
  const baselineSensitivity = baseRange.map(base => {
    const totalN = Math.round(((samplesPerDayMin + samplesPerDayMax) / 2) * ((durationMin + durationMax) / 2));
    const n = Math.max(10, Math.round(totalN / 2));
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = base * (effectMin + effectMax) / 200;
    const muA = base, muB = base + effect;
    let freqSig = 0, pbaSum = 0;
    for (let t = 0; t < nTrials; t++) {
      const xA = sampleMean(muA, sigma, n);
      const xB = sampleMean(muB, sigma, n);
      const { p } = tTest(xA, xB, sigma, sigma, n, n);
      if (p < 0.05) freqSig++;
      const postA = bayesianUpdate(priorMean, priorStrength, priorVar, xA, sigma, n);
      const postB = bayesianUpdate(priorMean + effect, priorStrength, priorVar, xB, sigma, n);
      pbaSum += probBGreaterA(postA, postB, 300);
    }
    return {
      baseline: base.toFixed(2),
      freqPower: freqSig / nTrials,
      pba: pbaSum / nTrials,
      effect: effect.toFixed(3),
    };
  });

  // ── Prior mean sensitivity ──
  const priorMeanSensitivity = priorMeanRange.map(pm => {
    const totalN = Math.round(((samplesPerDayMin + samplesPerDayMax) / 2) * ((durationMin + durationMax) / 2));
    const n = Math.max(10, Math.round(totalN / 2));
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = baselineMean * (effectMin + effectMax) / 200;
    const muA = baselineMean, muB = baselineMean + effect;
    const kappas = [1, 10, 50];
    const result = { priorMean: pm.toFixed(3) };
    for (const k of kappas) {
      let pbaSum = 0;
      for (let t = 0; t < 50; t++) {
        const xA = sampleMean(muA, sigma, n);
        const xB = sampleMean(muB, sigma, n);
        const postA = bayesianUpdate(pm, k, priorVar, xA, sigma, n);
        const postB = bayesianUpdate(pm + effect, k, priorVar, xB, sigma, n);
        pbaSum += probBGreaterA(postA, postB, 200);
      }
      result[`k${k}`] = pbaSum / 50;
    }
    return result;
  });

  // ── Prior strength vs duration heatmap ──
  const priorStrHeatmap = (() => {
    const rows = [];
    const kappas = [1, 5, 20, 50, 100];
    const durations = [7, 14, 30, 60, 90];
    const spd = Math.round((samplesPerDayMin + samplesPerDayMax) / 2);
    const sigma = (sigmaMin + sigmaMax) / 2;
    const effect = baselineMean * (effectMin + effectMax) / 200;
    const muA = baselineMean, muB = baselineMean + effect;
    for (const k of kappas) {
      for (const dur of durations) {
        const n = Math.max(10, spd * dur);
        let confident = 0;
        const M = 60;
        for (let t = 0; t < M; t++) {
          const xA = sampleMean(muA, sigma, n);
          const xB = sampleMean(muB, sigma, n);
          const postA = bayesianUpdate(priorMean, k, priorVar, xA, sigma, n);
          const postB = bayesianUpdate(priorMean + effect, k, priorVar, xB, sigma, n);
          const pba = probBGreaterA(postA, postB, 200);
          if (pba > 0.95 || pba < 0.05) confident++;
        }
        rows.push({ kappa: k, duration: dur, confidence: confident / M });
      }
    }
    return rows;
  })();

  return {
    powerVsSampleSize,
    pValueDist,
    pbaTraj,
    effectEstimation,
    concordance,
    fdrVsN,
    baselineSensitivity,
    priorMeanSensitivity,
    priorStrHeatmap,
  };
}

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function RangeSlider({ label, min, max, step, valueMin, valueMax, onChange, unit = "", description }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: C.accent1, fontSize: 12, fontFamily: "monospace", letterSpacing: 1 }}>{label}</span>
        <span style={{ color: C.accent3, fontSize: 12, fontFamily: "monospace" }}>
          [{valueMin}{unit} → {valueMax}{unit}]
        </span>
      </div>
      {description && <div style={{ color: C.muted, fontSize: 11, marginBottom: 6 }}>{description}</div>}
      {/* Color legend */}
      <div style={{ display: "flex", gap: 14, marginBottom: 5 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: C.accent1 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: C.accent1, display: "inline-block" }} />
          Lower bound
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: C.accent2 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: C.accent2, display: "inline-block" }} />
          Upper bound
        </span>
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <span style={{ color: C.muted, fontSize: 11, width: 40 }}>{min}{unit}</span>
        <div style={{ flex: 1, position: "relative", height: 20 }}>
          <input type="range" min={min} max={max} step={step} value={valueMin}
            onChange={e => onChange([parseFloat(e.target.value), valueMax])}
            style={{ position: "absolute", width: "100%", accentColor: C.accent1, cursor: "pointer" }} />
          <input type="range" min={min} max={max} step={step} value={valueMax}
            onChange={e => onChange([valueMin, parseFloat(e.target.value)])}
            style={{ position: "absolute", width: "100%", accentColor: C.accent2, cursor: "pointer", top: 8 }} />
        </div>
        <span style={{ color: C.muted, fontSize: 11, width: 40, textAlign: "right" }}>{max}{unit}</span>
      </div>
    </div>
  );
}

function SingleSlider({ label, min, max, step, value, onChange, unit = "", description }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: C.accent4, fontSize: 12, fontFamily: "monospace", letterSpacing: 1 }}>{label}</span>
        <span style={{ color: C.accent3, fontSize: 12, fontFamily: "monospace" }}>{value}{unit}</span>
      </div>
      {description && <div style={{ color: C.muted, fontSize: 11, marginBottom: 6 }}>{description}</div>}
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: C.accent4, cursor: "pointer" }} />
    </div>
  );
}

function Card({ title, children, accent = C.accent1 }) {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
      padding: 20, marginBottom: 20,
      boxShadow: `0 0 20px ${accent}11`,
    }}>
      {title && (
        <div style={{
          color: accent, fontSize: 13, fontFamily: "monospace", letterSpacing: 2,
          textTransform: "uppercase", marginBottom: 16, borderBottom: `1px solid ${C.border}`,
          paddingBottom: 10,
        }}>{title}</div>
      )}
      {children}
    </div>
  );
}

function MetricPill({ label, value, color = C.accent1 }) {
  return (
    <div style={{
      background: `${color}15`, border: `1px solid ${color}40`, borderRadius: 8,
      padding: "8px 14px", textAlign: "center",
    }}>
      <div style={{ color: C.muted, fontSize: 10, letterSpacing: 1, marginBottom: 4 }}>{label}</div>
      <div style={{ color, fontSize: 18, fontFamily: "monospace", fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function HeatmapCell({ value, label }) {
  const r = Math.round(255 * (1 - value));
  const g = Math.round(255 * value);
  const bg = `rgb(${r},${g},80)`;
  return (
    <td style={{
      background: bg, width: 60, height: 40, textAlign: "center",
      fontSize: 10, color: "#000", fontWeight: 700, border: `1px solid ${C.bg}`,
    }}>
      {(value * 100).toFixed(0)}%
    </td>
  );
}

const getTooltipStyle = () => ({
  background: C.surface, border: `1px solid ${C.border}`,
  borderRadius: 8, color: C.text, fontSize: 11, fontFamily: "monospace",
});

// ─── ASSUMPTIONS SECTION ──────────────────────────────────────────────────────
function AssumptionsSection() {
  const [tab, setTab] = useState("assumptions");
  const tabs = ["assumptions", "frequentist", "bayesian", "monte carlo"];

  const content = {
    assumptions: [
      { icon: "📐", title: "Metric Distribution", text: "Observations modeled as Normal(μ, σ²). Justified by CLT for large n (n > 30 per group recommended). Violations occur with heavy-tailed metrics like revenue." },
      { icon: "🔀", title: "Independence", text: "Each user/trial is i.i.d. across days and groups. Network effects, carryover, and SUTVA violations are NOT modeled. In practice, cluster randomization may be needed." },
      { icon: "📅", title: "No Temporal Effects", text: "Treatment effect is stable over the experiment window. Novelty effects, seasonality, and ramp-up are excluded. Consider holdout validation in production.", warn: true },
      { icon: "🔒", title: "Fixed Sample Size (Frequentist)", text: "P-values are valid only when sample size is fixed in advance. Peeking and early stopping inflate Type I error to ~20–30% even with α=0.05.", warn: true },
      { icon: "🔔", title: "Conjugate Prior Family", text: "Normal-Normal conjugate for continuous metrics. This yields closed-form posterior updates. Beta-Binomial is standard for conversion rates (binary outcomes)." },
      { icon: "⚖️", title: "Significance Threshold", text: "α = 0.05 (two-tailed), power target = 0.80. These are industry conventions, not statistical laws. High-stakes decisions warrant α = 0.01 or Bonferroni correction for multiple tests." },
    ],
    frequentist: [
      { icon: "📊", title: "Two-Sample t-Statistic", formula: "t = (X̄_B − X̄_A) / √(2σ²/n)", text: "Measures how many standard errors separate the group means. Assumes equal variance and equal sample sizes." },
      { icon: "🎯", title: "P-value", formula: "p = 2 × (1 − CDF_t(|t|, df=2n−2))", text: "Probability of observing a test statistic at least as extreme as |t| under the null hypothesis. NOT the probability that H₀ is true." },
      { icon: "📏", title: "Required Sample Size", formula: "n = 2 × ((z_α/2 + z_β)² × σ²) / δ²", text: "Minimum n per group for z_α/2 = 1.96 (α=0.05), z_β = 0.842 (power=80%). Increases quadratically as δ shrinks." },
      { icon: "💪", title: "Statistical Power", formula: "Power = P(reject H₀ | H₁ true) = 1 − β", text: "Empirically computed as the fraction of simulation trials where p < 0.05 when a true effect exists." },
      { icon: "📉", title: "Confidence Interval", formula: "CI = (X̄_B − X̄_A) ± t* × SE", text: "95% CI does NOT mean 95% probability the true effect lies within it. It means 95% of such intervals from repeated experiments contain the true effect." },
      { icon: "🚨", title: "False Discovery Rate", formula: "FDR = FP / (FP + TP)", text: "Share of 'significant' results that are false positives. Increases with low prevalence of true effects and low power.", warn: true },
    ],
    bayesian: [
      { icon: "🧠", title: "Prior", formula: "μ ~ N(μ_prior, σ²_prior / κ₀)", text: "Encodes belief about the true treatment effect before seeing data. κ₀ = prior strength (pseudo-count). Higher κ₀ resists updating from data." },
      { icon: "🔄", title: "Posterior Update", formula: "μ_post = (κ₀×μ_prior + n×X̄) / (κ₀ + n)", text: "Posterior mean is a weighted average of prior and data. As n >> κ₀, data dominates. As κ₀ >> n, prior dominates." },
      { icon: "📦", title: "Posterior Variance", formula: "σ²_post = σ² / (κ₀ + n)", text: "Posterior uncertainty shrinks as more data accumulates, regardless of what the data shows." },
      { icon: "🏆", title: "P(B > A)", formula: "P(μ_B − μ_A > 0) via Monte Carlo sampling", text: "Direct probability that treatment is better than control. Intuitive and decision-relevant. NOT affected by peeking in the same way as p-values." },
      { icon: "⚠️", title: "Prior Dominance Zone", formula: "When κ₀ > n: prior outweighs data", text: "In this region, two analysts with different priors will reach different conclusions from identical data. Results are non-replicable.", warn: true },
      { icon: "💰", title: "Expected Loss", formula: "E[Loss] = E[max(μ_A − μ_B, 0)]", text: "Expected cost of choosing B when A is actually better. Decision-theoretic stopping criterion. Stop when E[Loss] < business threshold." },
    ],
    "monte carlo": [
      { icon: "🎲", title: "Trials per Config Point", formula: "N = 150 simulated experiments", text: "Each parameter sweep point runs 150 independent simulated experiments. Empirical power = fraction significant. Reduce for speed, increase for precision." },
      { icon: "📍", title: "Sweep Resolution", formula: "10 evenly spaced points per range", text: "Each input range is divided into 10 points. Charts show how outcomes vary across this grid. Edge effects may appear at extremes." },
      { icon: "🔢", title: "Posterior Sampling", formula: "2000 draws from posterior per P(B>A) estimate", text: "Monte Carlo integration of P(μ_B > μ_A). Accuracy ~1/√2000 ≈ 2.2%. Sufficient for visualization; increase for production decisions." },
      { icon: "📐", title: "Normal Approximation", formula: "CLT + Normal-Normal conjugate", text: "Bypasses MCMC for speed. Valid for unimodal, symmetric posteriors. For skewed metrics (LTV, revenue), use log-normal or bootstrap." },
      { icon: "🌱", title: "Random Seed", text: "Each simulation run uses a different random seed. Re-run to see natural sampling variability — this itself is a lesson about Monte Carlo uncertainty." },
    ],
  };

  return (
    <Card title="⚙ Assumptions & Calculations" accent={C.accent5}>
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? `${C.accent5}30` : "transparent",
            border: `1px solid ${tab === t ? C.accent5 : C.border}`,
            color: tab === t ? C.accent5 : C.muted,
            padding: "6px 16px", borderRadius: 20, cursor: "pointer",
            fontSize: 12, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: 1,
          }}>{t}</button>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
        {content[tab].map((item, i) => (
          <div key={i} style={{
            background: item.warn ? `${C.accent2}08` : `${C.accent5}08`,
            border: `1px solid ${item.warn ? C.accent2 + "40" : C.border}`,
            borderRadius: 8, padding: 14,
          }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 6, alignItems: "center" }}>
              <span style={{ fontSize: 16 }}>{item.icon}</span>
              <span style={{ color: item.warn ? C.accent2 : C.accent4, fontSize: 12, fontWeight: 700 }}>{item.title}</span>
              {item.warn && <span style={{ color: C.accent2, fontSize: 10, marginLeft: "auto" }}>⚠ CAUTION</span>}
            </div>
            {item.formula && (
              <div style={{
                background: C.bg, borderRadius: 6, padding: "6px 10px",
                fontFamily: "monospace", fontSize: 11, color: C.accent3, marginBottom: 8,
              }}>{item.formula}</div>
            )}
            <div style={{ color: C.muted, fontSize: 11, lineHeight: 1.6 }}>{item.text}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ─── APP ──────────────────────────────────────────────────────────────────────
export default function App() {
  const SIM_TRIALS = 150;
  const [darkMode, setDarkMode] = useState(true);
  C = makeTheme(darkMode); // update module-level theme on every render

  const [params, setParams] = useState({
    samplesPerDayMin: 200, samplesPerDayMax: 2000,
    durationMin: 7, durationMax: 60,
    sigmaMin: 0.5, sigmaMax: 3.0,
    effectMin: 2, effectMax: 15,
    baselineMean: 10,
    priorMean: 0,
    priorStrength: 10,
    priorVar: 1.0,
  });

  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("power");
  const [showScorecardDefs, setShowScorecardDefs] = useState(false);
  const [showAssumptionsSection, setShowAssumptionsSection] = useState(true);

  const set = useCallback((key) => (val) => {
    if (Array.isArray(val)) {
      const [mn, mx] = val;
      setParams(p => ({ ...p, [`${key}Min`]: mn, [`${key}Max`]: mx }));
    } else {
      setParams(p => ({ ...p, [key]: val }));
    }
  }, []);

  const runSim = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const res = runSimulation(params, SIM_TRIALS);
      setResults(res);
      setRunning(false);
    }, 50);
  }, [params]);

  const chartTabs = [
    { id: "power", label: "Power" },
    { id: "pvalue", label: "P-Values" },
    { id: "pba", label: "P(B>A)" },
    { id: "fdr", label: "FDR" },
    { id: "effect", label: "Effect Est." },
    { id: "concordance", label: "Concordance" },
    { id: "sensitivity", label: "Sensitivity" },
  ];

  const avgDurationMid = (params.durationMin + params.durationMax) / 2;
  const avgDurationLabel = Number.isInteger(avgDurationMid) ? avgDurationMid.toString() : avgDurationMid.toFixed(1);
  const spdMid = Math.round((params.samplesPerDayMin + params.samplesPerDayMax) / 2);
  const totalSamplesMid = Math.round(spdMid * avgDurationMid);
  const perVariantSamplesMid = Math.max(10, Math.round(totalSamplesMid / 2));
  const totalSamplesMinBound = Math.round(params.samplesPerDayMin * avgDurationMid);
  const totalSamplesMaxBound = Math.round(params.samplesPerDayMax * avgDurationMid);

  const summaryMetrics = useMemo(() => {
    if (!results) return null;
    const last = results.powerVsSampleSize[results.powerVsSampleSize.length - 1];
    const mid = results.pbaTraj[Math.floor(results.pbaTraj.length / 2)];
    const lastFDR = results.fdrVsN[results.fdrVsN.length - 1];
    return {
      freqPower: (last.freqPower * 100).toFixed(1) + "%",
      bayesPower: (last.bayesPower * 100).toFixed(1) + "%",
      pba: (mid.pba * 100).toFixed(1) + "%",
      pbaMidDay: mid.day,
      freqFDR: (lastFDR.freqFDR * 100).toFixed(1) + "%",
      bayesFDR: (lastFDR.bayesFDR * 100).toFixed(1) + "%",
      requiredN: last.required === Infinity ? "∞" : last.required.toLocaleString(),
    };
  }, [results]);

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: "'Courier New', monospace", padding: 24,
      transition: "background 0.3s, color 0.3s",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 32, position: "relative" }}>
        {/* Dark / Light toggle */}
        <button
          onClick={() => setDarkMode(d => !d)}
          title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          style={{
            position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)",
            background: darkMode ? "#1e3a5f" : "#e2e8f0",
            border: `1px solid ${C.border}`,
            borderRadius: 20, padding: "6px 14px", cursor: "pointer",
            display: "flex", alignItems: "center", gap: 7,
            fontSize: 12, fontFamily: "monospace", color: C.text,
            transition: "all 0.2s",
          }}
        >
          <span style={{ fontSize: 15 }}>{darkMode ? "☀️" : "🌙"}</span>
          {darkMode ? "LIGHT" : "DARK"}
        </button>

        <div style={{
          fontSize: 28, fontWeight: 900, letterSpacing: 4,
          color: C.text,
          marginBottom: 6,
        }}>A/B TEST SIMULATOR</div>
        <div style={{ color: C.muted, fontSize: 12, letterSpacing: 3 }}>
          FREQUENTIST × BAYESIAN · MONTE CARLO ENGINE
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20, maxWidth: 1400, margin: "0 auto" }}>
        {/* Left panel — Inputs */}
        <div>
          <Card title="▸ Experiment Parameters" accent={C.accent1}>
            <RangeSlider label="SAMPLES / DAY" min={50} max={5000} step={50}
              valueMin={params.samplesPerDayMin} valueMax={params.samplesPerDayMax}
              onChange={set("samplesPerDay")} description="Daily traffic per variant" />
            <RangeSlider label="DURATION (DAYS)" min={3} max={90} step={1}
              valueMin={params.durationMin} valueMax={params.durationMax}
              onChange={set("duration")} description="Experiment window length" />
            <RangeSlider label="METRIC STD DEV (σ)" min={0.1} max={10} step={0.1}
              valueMin={params.sigmaMin} valueMax={params.sigmaMax}
              onChange={set("sigma")} description="Noise level in the metric" />
            <RangeSlider label="DESIRED EFFECT (MDE)" min={0.5} max={30} step={0.5}
              valueMin={params.effectMin} valueMax={params.effectMax}
              onChange={set("effect")} unit="%" description="Minimum detectable effect as % of baseline" />
          </Card>

          <Card title="▸ Baseline & Metric" accent={C.accent4}>
            <SingleSlider label="BASELINE MEAN (μ₀)" min={0.1} max={100} step={0.1}
              value={params.baselineMean} onChange={set("baselineMean")}
              description="Control group expected value (e.g. 5 = 5% CVR or $5 AOV)" />
          </Card>

          <Card title="▸ Bayesian Prior" accent={C.accent5}>
            <SingleSlider label="PRIOR MEAN ON EFFECT" min={-5} max={5} step={0.1}
              value={params.priorMean} onChange={set("priorMean")}
              description="Prior belief about treatment effect. 0 = no effect expected (conservative)" />
            <SingleSlider label="PRIOR STRENGTH (κ₀)" min={1} max={200} step={1}
              value={params.priorStrength} onChange={set("priorStrength")}
              description="Pseudo-sample-count for prior. High = resists data. Low = diffuse prior" />
            <SingleSlider label="PRIOR VARIANCE (σ²_prior)" min={0.1} max={10} step={0.1}
              value={params.priorVar} onChange={set("priorVar")}
              description="Uncertainty around prior mean. Larger = more diffuse prior" />
            <div style={{
              background: params.priorStrength > 50 ? `${C.accent2}20` : `${C.accent5}10`,
              border: `1px solid ${params.priorStrength > 50 ? C.accent2 : C.accent5}40`,
              borderRadius: 8, padding: 10, fontSize: 11, color: C.muted, marginTop: 8,
            }}>
              {params.priorStrength > 50
                ? `⚠ Strong prior (κ₀=${params.priorStrength}). Prior dominates data when n < ${params.priorStrength * 2}. Results may be non-replicable.`
                : `Prior strength κ₀=${params.priorStrength}. Data dominates when n > ${params.priorStrength * 5}.`
              }
            </div>
          </Card>

          <button onClick={runSim} disabled={running} style={{
            width: "100%", padding: "14px 0",
            background: `${C.accent1}22`,
            border: `1px solid ${C.accent1}`,
            color: C.text, borderRadius: 10,
            fontSize: 14, fontFamily: "monospace", letterSpacing: 3,
            cursor: running ? "not-allowed" : "pointer",
            transition: "all 0.2s",
          }}>
            {running ? `⏱  RUNNING ${SIM_TRIALS} MONTE CARLO TRIALS / POINT...` : "▶  RUN SIMULATION"}
          </button>
          <div style={{ marginTop: 8, color: C.muted, fontSize: 10 }}>
            {running
              ? `Running ${SIM_TRIALS} Monte Carlo simulations per sweep point.`
              : `Will run ${SIM_TRIALS} Monte Carlo simulations per sweep point.`}
          </div>
        </div>

        {/* Right panel — Results */}
        <div>
          {!results && !running && (
            <div style={{
              height: 400, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              border: `1px dashed ${C.border}`, borderRadius: 12,
              color: C.muted, gap: 16,
            }}>
              <div style={{ fontSize: 48 }}>◈</div>
              <div style={{ fontSize: 13, letterSpacing: 2 }}>SET PARAMETERS AND RUN SIMULATION</div>
              <div style={{ fontSize: 11, color: C.border }}>150 Monte Carlo trials × 10 sweep points</div>
            </div>
          )}

          {running && (
            <div style={{
              height: 400, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              border: `1px solid ${C.border}`, borderRadius: 12, gap: 20,
            }}>
              <div style={{
                width: 60, height: 60, border: `3px solid ${C.border}`,
                borderTopColor: C.accent1, borderRadius: "50%",
                animation: "spin 1s linear infinite",
              }} />
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              <div style={{ color: C.accent1, fontSize: 13, letterSpacing: 2 }}>RUNNING MONTE CARLO...</div>
              <div style={{ color: C.muted, fontSize: 11 }}>Computing 150 trials × 10 parameter sweep points</div>
            </div>
          )}

          {results && summaryMetrics && (
            <>
              {/* Summary scorecard */}
              <Card title="▸ Summary Scorecard" accent={C.accent3}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
                  <div>
                    <div style={{ color: C.muted, fontSize: 10, marginBottom: 8, letterSpacing: 2 }}>FREQUENTIST</div>
                    <div style={{ display: "grid", gap: 8 }}>
                      <MetricPill label="Power (max n)" value={summaryMetrics.freqPower} color={C.accent1} />
                      <MetricPill label="FDR (max n)" value={summaryMetrics.freqFDR} color={C.accent2} />
                      <MetricPill label="Required n" value={summaryMetrics.requiredN} color={C.accent3} />
                    </div>
                  </div>
                  <div>
                    <div style={{ color: C.muted, fontSize: 10, marginBottom: 8, letterSpacing: 2 }}>BAYESIAN</div>
                    <div style={{ display: "grid", gap: 8 }}>
                      <MetricPill label="Decision Rate" value={summaryMetrics.bayesPower} color={C.accent4} />
                      <MetricPill label="False Conf." value={summaryMetrics.bayesFDR} color={C.accent2} />
                      <MetricPill label={`Mean P(B>A) @ day ${summaryMetrics.pbaMidDay}`} value={summaryMetrics.pba} color={C.accent5} />
                    </div>
                  </div>
                  <div>
                    <div style={{ color: C.muted, fontSize: 10, marginBottom: 8, letterSpacing: 2 }}>COMPARISON</div>
                    <div style={{ display: "grid", gap: 8 }}>
                      <div style={{
                        background: parseFloat(summaryMetrics.bayesPower) > parseFloat(summaryMetrics.freqPower)
                          ? `${C.win}15` : `${C.accent3}15`,
                        border: `1px solid ${C.border}`, borderRadius: 8, padding: "8px 14px", textAlign: "center",
                      }}>
                        <div style={{ color: C.muted, fontSize: 10, marginBottom: 4 }}>FASTER DECISION</div>
                        <div style={{ color: C.accent4, fontSize: 14, fontWeight: 700 }}>
                          {parseFloat(summaryMetrics.bayesPower) > parseFloat(summaryMetrics.freqPower) ? "BAYESIAN" : "FREQUENTIST"}
                        </div>
                      </div>
                      <div style={{
                        background: `${C.accent5}15`, border: `1px solid ${C.border}`,
                        borderRadius: 8, padding: "8px 14px", textAlign: "center",
                      }}>
                        <div style={{ color: C.muted, fontSize: 10, marginBottom: 4 }}>LOWER ERROR</div>
                        <div style={{ color: C.accent5, fontSize: 14, fontWeight: 700 }}>
                          {parseFloat(summaryMetrics.bayesFDR) < parseFloat(summaryMetrics.freqFDR) ? "BAYESIAN" : "FREQUENTIST"}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
              <div style={{ marginTop: 10, marginBottom: 14 }}>
                <button
                  onClick={() => setShowScorecardDefs(v => !v)}
                  style={{
                    background: "transparent",
                    border: `1px solid ${C.border}`,
                    color: C.muted,
                    borderRadius: 8,
                    padding: "6px 10px",
                    fontSize: 11,
                    fontFamily: "monospace",
                    cursor: "pointer",
                  }}
                >
                  {showScorecardDefs ? "▾ Hide scorecard definitions" : "▸ Show scorecard definitions"}
                </button>
                {showScorecardDefs && (
                  <div style={{
                    marginTop: 8, border: `1px solid ${C.border}`, borderRadius: 8,
                    padding: 10, background: C.surface, color: C.muted, fontSize: 11, lineHeight: 1.55,
                  }}>
                    <div><strong style={{ color: C.text }}>Power (max n):</strong> Frequentist probability of detecting a real effect at the largest tested sample size.</div>
                    <div><strong style={{ color: C.text }}>FDR (max n):</strong> False Discovery Rate, i.e. false positives divided by all positive decisions at max n.</div>
                    <div><strong style={{ color: C.text }}>Required n:</strong> Approximate total sample size needed for 80% power under current assumptions.</div>
                    <div><strong style={{ color: C.text }}>Decision Rate:</strong> Bayesian share of trials reaching the confidence threshold P(B&gt;A) &gt; 0.95 at max n.</div>
                    <div><strong style={{ color: C.text }}>False Conf.:</strong> Bayesian false-confidence rate, analogous to FDR for confident Bayesian calls.</div>
                    <div><strong style={{ color: C.text }}>Mean P(B&gt;A) @ midpoint day:</strong> Mean posterior probability at the midpoint day on the P(B&gt;A) chart (not the max value).</div>
                    <div><strong style={{ color: C.text }}>Faster Decision:</strong> Method with higher decision/power rate under current settings.</div>
                    <div><strong style={{ color: C.text }}>Lower Error:</strong> Method with lower false-positive/false-confidence rate under current settings.</div>
                  </div>
                )}
              </div>

              {/* Chart tabs */}
              <Card title="▸ Charts" accent={C.accent1}>
                <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
                  {chartTabs.map(t => (
                    <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
                      background: activeTab === t.id ? `${C.accent1}25` : "transparent",
                      border: `1px solid ${activeTab === t.id ? C.accent1 : C.border}`,
                      color: activeTab === t.id ? C.accent1 : C.muted,
                      padding: "5px 14px", borderRadius: 20, cursor: "pointer",
                      fontSize: 11, fontFamily: "monospace",
                    }}>{t.label}</button>
                  ))}
                </div>

                {/* Power chart */}
                {activeTab === "power" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 12 }}>
                      Statistical power across total sample sizes. Target: 80% (dashed). Bayesian "power" = P(B{">"} A) {">"} 0.95.
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={results.powerVsSampleSize} margin={{ top: 10, right: 130, bottom: 28, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="sampleSize" stroke={C.muted} tick={{ fontSize: 10 }} label={{ value: "Total Sample Size (n)", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={58} stroke={C.muted} tick={{ fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} label={{ value: "Power / Decision Rate", angle: -90, position: "insideLeft", dx: -22, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip contentStyle={getTooltipStyle()} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: 10, lineHeight: "18px" }} />
                        <ReferenceLine y={0.8} stroke={C.accent3} strokeDasharray="6 3" label={{ value: "80% target", fill: C.accent3, fontSize: 10 }} />
                        <Line dataKey="freqPower" name="Frequentist" stroke={C.accent1} strokeWidth={2} dot={false} />
                        <Line dataKey="bayesPower" name="Bayesian" stroke={C.accent4} strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 6 }}>
                      Footnote: Total sample size n is computed as samples/day × average duration, where average duration = (duration min + duration max) / 2. In this run, avg duration = {avgDurationLabel} days, so x-axis bounds are {params.samplesPerDayMin} × {avgDurationLabel} = {totalSamplesMinBound} and {params.samplesPerDayMax} × {avgDurationLabel} = {totalSamplesMaxBound}. Midpoint example: {spdMid} × {avgDurationLabel} = {totalSamplesMid}. Per-variant sample size used in tests is approx n/2 (midpoint: {perVariantSamplesMid} per arm).
                    </div>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 8, padding: 10, background: `${C.accent3}10`, borderRadius: 6, border: `1px solid ${C.accent3}30` }}>
                      💡 <strong style={{ color: C.accent3 }}>Insight:</strong> The S-curve transition from underpowered to overpowered region shows where your experiment crosses the 80% power threshold. Points left of this are unreliable — you'll miss true effects 20%+ of the time.
                    </div>
                  </div>
                )}

                {/* P-value distribution */}
                {activeTab === "pvalue" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 12 }}>
                      P-value histogram under true effect (blue) vs null hypothesis (red). Under null, distribution should be uniform.
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={results.pValueDist} margin={{ top: 10, right: 130, bottom: 20, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="bucket" stroke={C.muted} tick={{ fontSize: 9 }} label={{ value: "P-value Bucket", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={58} stroke={C.muted} tick={{ fontSize: 10 }} label={{ value: "Trial Count", angle: -90, position: "insideLeft", dx: -22, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip contentStyle={getTooltipStyle()} />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: 10, lineHeight: "18px" }} />
                        <Bar dataKey="withEffect" name="True Effect" fill={C.accent1} opacity={0.8} />
                        <Bar dataKey="nullHypothesis" name="Null (H0)" fill={C.accent2} opacity={0.8} />
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 8, padding: 10, background: `${C.accent2}10`, borderRadius: 6, border: `1px solid ${C.accent2}30` }}>
                      ⚠ <strong style={{ color: C.accent2 }}>Warning:</strong> Under H₀, p-values are uniformly distributed — every bucket has equal probability. The spike near p=0 under true effect is power. If your null distribution is NOT uniform, your randomization is broken.
                    </div>
                  </div>
                )}

                {/* P(B>A) trajectory */}
                {activeTab === "pba" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 12 }}>
                      Bayesian posterior P(B{">"} A) evolving over experiment duration. Band shows trial-to-trial variability (min/max across 30 trials).
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={results.pbaTraj} margin={{ top: 10, right: 20, bottom: 20, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="day" stroke={C.muted} tick={{ fontSize: 10 }} label={{ value: "Experiment Day", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={58} stroke={C.muted} tick={{ fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} label={{ value: "P(B > A)", angle: -90, position: "insideLeft", dx: -22, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip
                          contentStyle={getTooltipStyle()}
                          formatter={(v, n) => {
                            const val = `${(v * 100).toFixed(1)}%`;
                            if (n === "Mean P(B>A)" || n === "Max P(B>A)" || n === "Min P(B>A)") return [val, n];
                            return [val, n];
                          }}
                        />
                        <ReferenceLine y={0.95} stroke={C.win} strokeDasharray="6 3" label={{ value: "95% threshold", fill: C.win, fontSize: 10 }} />
                        <ReferenceLine y={0.5} stroke={C.muted} strokeDasharray="3 3" />
                        <Line dataKey="high" name="Max P(B>A)" stroke={C.accent4} strokeOpacity={0.9} strokeWidth={1.6} strokeDasharray="4 3" dot={false} />
                        <Line dataKey="low" name="Min P(B>A)" stroke={C.accent2} strokeOpacity={0.9} strokeWidth={1.6} strokeDasharray="4 3" dot={false} />
                        <Line dataKey="pba" name="Mean P(B>A)" stroke={C.accent1} strokeWidth={3} dot={{ r: 2 }} activeDot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 8, padding: 10, background: `${C.accent4}10`, borderRadius: 6, border: `1px solid ${C.accent4}30` }}>
                      💡 <strong style={{ color: C.accent4 }}>Insight:</strong> Unlike frequentist p-values, you can monitor P(B{">"} A) daily without inflating error rates. The Bayesian framework naturally incorporates sequential evidence — but strong priors can pull the trajectory away from the data early on.
                    </div>
                  </div>
                )}

                {/* FDR */}
                {activeTab === "fdr" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 12 }}>
                      False Discovery Rate as a function of total sample size. Lower is better. FDR = false positives / all positives.
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={results.fdrVsN} margin={{ top: 10, right: 130, bottom: 20, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="sampleSize" stroke={C.muted} tick={{ fontSize: 10 }} label={{ value: "Total Sample Size (n)", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={58} stroke={C.muted} tick={{ fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} label={{ value: "False Discovery Rate", angle: -90, position: "insideLeft", dx: -22, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip contentStyle={getTooltipStyle()} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: 10, lineHeight: "18px" }} />
                        <Line dataKey="freqFDR" name="Freq. FDR" stroke={C.accent1} strokeWidth={2} dot={false} />
                        <Line dataKey="bayesFDR" name="Bayes. False Conf." stroke={C.accent5} strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 6 }}>
                      Footnote: For each point, total samples are n = samples/day × {avgDurationLabel}, where avg duration = (duration min + duration max) / 2. This yields bounds {totalSamplesMinBound} to {totalSamplesMaxBound}; tests use approx n/2 per variant (50/50 split).
                    </div>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 8, padding: 10, background: `${C.accent2}10`, borderRadius: 6, border: `1px solid ${C.accent2}30` }}>
                      ⚠ <strong style={{ color: C.accent2 }}>Warning:</strong> FDR is highest at small sample sizes — this is the "winner's curse" region where significant results are likely false positives. Both methods converge toward low FDR with large n, but Bayesian methods with strong priors can maintain low false confidence even at small n.
                    </div>
                  </div>
                )}

                {/* Effect estimation */}
                {activeTab === "effect" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 12 }}>
                      Estimated vs true effect size. X-axis is % lift vs baseline; Y-axis is absolute effect in metric units (not %). Error bars show 5th–95th percentile. Bias = systematic over/underestimation.
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={results.effectEstimation} margin={{ top: 10, right: 130, bottom: 20, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="trueEffect" stroke={C.muted} tick={{ fontSize: 10 }} label={{ value: "True Effect Size", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={66} stroke={C.muted} tick={{ fontSize: 10 }} label={{ value: "Estimated Effect (metric units)", angle: -90, position: "insideLeft", dx: -26, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip contentStyle={getTooltipStyle()} formatter={(v) => typeof v === "number" ? v.toFixed(4) : v} />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: 10, lineHeight: "18px" }} />
                        <Line dataKey="p95" name="P95" stroke={C.accent2} strokeWidth={1} strokeDasharray="4 2" dot={false} />
                        <Line dataKey="p75" name="P75" stroke={C.accent3} strokeWidth={1} dot={false} />
                        <Line dataKey="estimatedMean" name="Mean" stroke={C.accent1} strokeWidth={2.5} dot={false} />
                        <Line dataKey="trueVal" name="True" stroke={C.accent4} strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
                        <Line dataKey="p25" name="P25" stroke={C.accent3} strokeWidth={1} dot={false} />
                        <Line dataKey="p5" name="P5" stroke={C.accent2} strokeWidth={1} strokeDasharray="4 2" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 8, padding: 10, background: `${C.accent3}10`, borderRadius: 6, border: `1px solid ${C.accent3}30` }}>
                      💡 <strong style={{ color: C.accent3 }}>Insight:</strong> With small n, the "winner's curse" means significant results overestimate the true effect. The estimated mean above/below the true line is systematic bias. Larger experiments bring the bands closer to the true line.
                    </div>
                  </div>
                )}

                {/* Concordance heatmap */}
                {activeTab === "concordance" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 12 }}>
                      % of trials where frequentist and Bayesian decisions agree. Green = high agreement, red = divergence.
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
                        <thead>
                          <tr>
                            <th style={{ color: C.muted, fontSize: 11, padding: "8px 12px", textAlign: "left" }}>n \ Effect</th>
                            {["1%", "5%", "10%", "20%"].map(e => (
                              <th key={e} style={{ color: C.accent1, fontSize: 11, padding: "8px 16px" }}>{e}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[100, 500, 1000, 5000].map(n => (
                            <tr key={n}>
                              <td style={{ color: C.accent4, fontSize: 11, padding: "8px 12px", fontWeight: 700 }}>n={n}</td>
                              {[1, 5, 10, 20].map(eff => {
                                const row = results.concordance.find(r => r.n === n.toString() && r.effect === `${eff}%`);
                                return row ? <HeatmapCell key={eff} value={row.concordance} /> : <td key={eff}>-</td>;
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 12, padding: 10, background: `${C.accent5}10`, borderRadius: 6, border: `1px solid ${C.accent5}30` }}>
                      💡 <strong style={{ color: C.accent5 }}>Insight:</strong> Divergence peaks at small n + small effect (top-left). This is where your prior dominates and Bayesian conclusions are driven by belief, not data. Large n + large effect (bottom-right) shows near-perfect agreement — both methods reach the same conclusion when evidence is overwhelming.
                    </div>
                  </div>
                )}

                {/* Sensitivity charts */}
                {activeTab === "sensitivity" && (
                  <div>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 16 }}>
                      How baseline mean and prior assumptions drive outcome — the "assumption sensitivity" analysis.
                    </div>

                    {/* Baseline sensitivity */}
                    <div style={{ marginBottom: 8, color: C.accent3, fontSize: 11, letterSpacing: 1 }}>BASELINE MEAN vs. OUTCOME</div>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={results.baselineSensitivity} margin={{ top: 10, right: 130, bottom: 20, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="baseline" stroke={C.muted} tick={{ fontSize: 9 }} label={{ value: "Baseline Mean (μ0)", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={58} stroke={C.muted} tick={{ fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} label={{ value: "Power / P(B > A)", angle: -90, position: "insideLeft", dx: -22, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip contentStyle={getTooltipStyle()} formatter={(v, n) => [`${(v * 100).toFixed(1)}%`, n]} />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: 10, lineHeight: "18px" }} />
                        <ReferenceLine y={0.8} stroke={C.accent3} strokeDasharray="4 2" />
                        <Line dataKey="freqPower" name="Freq. Power" stroke={C.accent1} strokeWidth={2} dot={false} />
                        <Line dataKey="pba" name="P(B>A)" stroke={C.accent4} strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>

                    {/* Prior mean sensitivity */}
                    <div style={{ marginTop: 20, marginBottom: 8, color: C.accent5, fontSize: 11, letterSpacing: 1 }}>PRIOR MEAN vs. P(B{">"} A) BY PRIOR STRENGTH</div>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={results.priorMeanSensitivity} margin={{ top: 10, right: 130, bottom: 20, left: 56 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="priorMean" stroke={C.muted} tick={{ fontSize: 9 }} label={{ value: "Prior Mean on Effect", position: "insideBottom", offset: -8, fill: C.muted, fontSize: 10 }} />
                        <YAxis width={58} stroke={C.muted} tick={{ fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} label={{ value: "P(B > A)", angle: -90, position: "insideLeft", dx: -22, fill: C.muted, fontSize: 10, style: { textAnchor: "middle" } }} />
                        <Tooltip contentStyle={getTooltipStyle()} formatter={(v, n) => [`${(v * 100).toFixed(1)}%`, n]} />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: 10, lineHeight: "18px" }} />
                        <Line dataKey="k1" name="κ₀=1" stroke={C.accent4} strokeWidth={2} dot={false} />
                        <Line dataKey="k10" name="κ₀=10" stroke={C.accent5} strokeWidth={2} dot={false} />
                        <Line dataKey="k50" name="κ₀=50" stroke={C.accent2} strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>

                    {/* Prior strength heatmap */}
                    <div style={{ marginTop: 20, marginBottom: 8, color: C.accent2, fontSize: 11, letterSpacing: 1 }}>PRIOR STRENGTH × DURATION → CONFIDENT DECISIONS (%)</div>
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
                        <thead>
                          <tr>
                            <th style={{ color: C.muted, fontSize: 11, padding: "6px 10px", textAlign: "left" }}>κ₀ \ Days</th>
                            {[7, 14, 30, 60, 90].map(d => (
                              <th key={d} style={{ color: C.accent1, fontSize: 11, padding: "6px 14px" }}>{d}d</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[1, 5, 20, 50, 100].map(k => (
                            <tr key={k}>
                              <td style={{ color: C.accent5, fontSize: 11, padding: "6px 10px", fontWeight: 700 }}>κ₀={k}</td>
                              {[7, 14, 30, 60, 90].map(d => {
                                const row = results.priorStrHeatmap.find(r => r.kappa === k && r.duration === d);
                                return row ? <HeatmapCell key={d} value={row.confidence} /> : <td key={d}>-</td>;
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div style={{ color: C.muted, fontSize: 10, marginTop: 10, padding: 10, background: `${C.accent2}10`, borderRadius: 6, border: `1px solid ${C.accent2}30` }}>
                      ⚠ <strong style={{ color: C.accent2 }}>Caution:</strong> Top-left zone (weak prior + short duration) is the danger quadrant — low confidence, high noise. Bottom-right (strong prior + long duration) reaches confident decisions fastest but risks prior dominance overriding data.
                    </div>
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Full-width assumptions section */}
      {results && (
        <div style={{ maxWidth: 1400, margin: "0 auto", marginTop: 18 }}>
          <button
            onClick={() => setShowAssumptionsSection(v => !v)}
            style={{
              background: "transparent",
              border: `1px solid ${C.border}`,
              color: C.muted,
              borderRadius: 8,
              padding: "6px 10px",
              fontSize: 11,
              fontFamily: "monospace",
              cursor: "pointer",
              marginBottom: showAssumptionsSection ? 10 : 0,
            }}
          >
            {showAssumptionsSection ? "▾ Hide assumptions & calculations" : "▸ Show assumptions & calculations"}
          </button>
          {showAssumptionsSection && <AssumptionsSection />}
        </div>
      )}
    </div>
  );
}
