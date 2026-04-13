import { useState } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid,
} from "recharts";

/* ═══════════════════════════════════════════════════════════════════
   CONFIG
═══════════════════════════════════════════════════════════════════ */
const API = "http://localhost:5000";


const BG_IMAGE = "/assets/churn-bg.png";

const DEFAULTS = {
  seniorCitizen: true, partner: true, dependents: true,
  paperlessBilling: false, phoneService: false, multipleLines: false,
  onlineSecurity: false, onlineBackup: false, deviceProtection: true,
  techSupport: false, streamingTV: false, streamingMovies: false,
  gender: "Male", internetService: "Fiber optic",
  contract: "Month-to-Month", paymentMethod: "Electronic Check",
  tenure: 40, monthlyCharges: 65,
};

const CHECKBOX_ROW1 = [
  { key: "seniorCitizen",    label: "Senior Citizen" },
  { key: "partner",          label: "Has Partner" },
  { key: "dependents",       label: "Dependents" },
  { key: "paperlessBilling", label: "Paperless Bill" },
  { key: "phoneService",     label: "Phone Service" },
  { key: "multipleLines",    label: "Multi Lines" },
];
const CHECKBOX_ROW2 = [
  { key: "onlineSecurity",   label: "Online Security" },
  { key: "onlineBackup",     label: "Online Backup" },
  { key: "deviceProtection", label: "Device Protect" },
  { key: "techSupport",      label: "Tech Support" },
  { key: "streamingTV",      label: "Streaming TV" },
  { key: "streamingMovies",  label: "Streaming Movies" },
];

const RISK = {
  LOW:     { color: "#30D158", label: "LOW RISK" },
  MEDIUM:  { color: "#FFD60A", label: "MED RISK" },
  HIGH:    { color: "#FF9F0A", label: "HIGH RISK" },
  EXTREME: { color: "#FF3B30", label: "CRITICAL" },
};

/* ═══════════════════════════════════════════════════════════════════
   GAUGE COMPONENT
═══════════════════════════════════════════════════════════════════ */
function Gauge({ prob, risk }) {
  const cfg = RISK[risk] || RISK.MEDIUM;
  const angle = -135 + prob * 270;
  const arc = (s, e, r, cx = 100, cy = 100) => {
    const rad = (d) => (d * Math.PI) / 180;
    const x1 = cx + r * Math.cos(rad(s)), y1 = cy + r * Math.sin(rad(s));
    const x2 = cx + r * Math.cos(rad(e)), y2 = cy + r * Math.sin(rad(e));
    return `M${x1} ${y1} A${r} ${r} 0 ${e - s > 180 ? 1 : 0} 1 ${x2} ${y2}`;
  };
  const segs = [
    { s: -135, e: -68, c: "#30D158" },
    { s: -68,  e:   0, c: "#FFD60A" },
    { s:   0,  e:  67, c: "#FF9F0A" },
    { s:  67,  e: 135, c: "#FF3B30" },
  ];
  return (
    <div style={{ textAlign: "center" }}>
      <svg width="200" height="145" viewBox="0 0 200 145">
        <path d={arc(-135, 135, 72)} fill="none" stroke="#1c1c1e" strokeWidth="16" strokeLinecap="round" />
        {segs.map((sg, i) => (
          <path key={i} d={arc(sg.s, sg.e, 72)} fill="none" stroke={sg.c}
            strokeWidth="16" strokeLinecap="round" opacity={0.85} />
        ))}
        <g transform={`rotate(${angle},100,100)`}>
          <line x1="100" y1="100" x2="100" y2="35"
            stroke="#f2f2f7" strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="100" cy="100" r="5" fill={cfg.color}
            style={{ filter: `drop-shadow(0 0 6px ${cfg.color})` }} />
        </g>
        <text x="22"  y="128" fill="#30D158" fontSize="8" fontFamily="monospace" fontWeight="700">LOW</text>
        <text x="54"  y="142" fill="#FFD60A" fontSize="8" fontFamily="monospace" fontWeight="700">MED</text>
        <text x="122" y="142" fill="#FF9F0A" fontSize="8" fontFamily="monospace" fontWeight="700">HIGH</text>
        <text x="152" y="128" fill="#FF3B30" fontSize="8" fontFamily="monospace" fontWeight="700">EXT</text>
      </svg>
      <div style={{ fontSize: 38, fontWeight: 900, color: cfg.color,
        fontFamily: "'Syne', sans-serif", letterSpacing: "-0.03em", lineHeight: 1,
        textShadow: `0 0 28px ${cfg.color}88` }}>
        {(prob * 100).toFixed(1)}%
      </div>
      <div style={{ fontSize: 9, letterSpacing: "0.22em", color: cfg.color,
        fontFamily: "monospace", marginTop: 5, opacity: 0.85 }}>{cfg.label}</div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SHAP BARS
═══════════════════════════════════════════════════════════════════ */
function ShapBars({ data }) {
  if (!data?.length) return (
    <div style={{ color: "#3a3a3c", textAlign: "center", padding: "32px 0",
      fontSize: 11, fontFamily: "monospace" }}>
      — SHAP values unavailable (model not loaded) —
    </div>
  );
  const max = Math.max(...data.map((d) => Math.abs(d.shap)));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      {data.slice(0, 10).map((item, i) => {
        const pct = (Math.abs(item.shap) / max) * 100;
        const pos = item.shap > 0;
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 175, fontSize: 10, color: "#8e8e93", fontFamily: "monospace",
              textAlign: "right", flexShrink: 0, overflow: "hidden",
              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {item.feature.replace(/_/g, " ")}
            </div>
            <div style={{ flex: 1, height: 14, background: "#1c1c1e",
              borderRadius: 3, overflow: "hidden", position: "relative" }}>
              <div style={{
                position: "absolute",
                left: pos ? "50%" : `${50 - pct / 2}%`,
                width: `${pct / 2}%`, height: "100%",
                background: pos ? "#FF3B30" : "#FF9F0A",
                borderRadius: pos ? "0 3px 3px 0" : "3px 0 0 3px",
              }} />
              <div style={{ position: "absolute", left: "50%", top: 0,
                width: 1, height: "100%", background: "#2c2c2e" }} />
            </div>
            <div style={{ width: 52, fontSize: 9, fontFamily: "monospace",
              flexShrink: 0, color: pos ? "#FF3B30" : "#FF9F0A" }}>
              {item.shap > 0 ? "+" : ""}{item.shap.toFixed(3)}
            </div>
          </div>
        );
      })}
      <div style={{ display: "flex", justifyContent: "center", gap: 20, marginTop: 8 }}>
        {[["#FF9F0A", "← Reduces churn"], ["#FF3B30", "Increases churn →"]].map(([c, l]) => (
          <div key={l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
            <span style={{ fontSize: 9, color: "#636366", fontFamily: "monospace" }}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   TOOLTIP
═══════════════════════════════════════════════════════════════════ */
const CT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#111", border: "1px solid #2c2c2e",
      borderRadius: 5, padding: "5px 10px", fontSize: 11, fontFamily: "monospace" }}>
      <div style={{ color: "#636366" }}>month {label}</div>
      <div style={{ color: "#FF9F0A" }}>{(payload[0].value * 100).toFixed(2)}%</div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   UI ATOMS
═══════════════════════════════════════════════════════════════════ */
function Chip({ checked, onChange, label }) {
  return (
    <div onClick={onChange} style={{
      padding: "6px 12px", borderRadius: 6, cursor: "pointer",
      userSelect: "none", fontSize: 11, fontFamily: "monospace",
      border: `1px solid ${checked ? "#FF9F0A66" : "#2c2c2e"}`,
      background: checked ? "#FF9F0A12" : "transparent",
      color: checked ? "#FF9F0A" : "#636366",
      transition: "all 0.15s",
      display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap",
    }}>
      <span style={{ fontSize: 7 }}>{checked ? "◉" : "◎"}</span>
      {label}
    </div>
  );
}

function Sel({ label, value, onChange, options }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: "#48484a", letterSpacing: "0.12em",
        textTransform: "uppercase", fontFamily: "monospace", marginBottom: 5 }}>{label}</div>
      <select value={value} onChange={(e) => onChange(e.target.value)} style={{
        width: "100%", background: "#1c1c1e", border: "1px solid #2c2c2e",
        borderRadius: 6, color: "#f2f2f7", padding: "7px 10px",
        fontSize: 12, outline: "none", cursor: "pointer", fontFamily: "monospace",
      }}>
        {options.map((o) => <option key={o}>{o}</option>)}
      </select>
    </div>
  );
}

function Num({ label, value, onChange, min, max }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: "#48484a", letterSpacing: "0.12em",
        textTransform: "uppercase", fontFamily: "monospace", marginBottom: 5 }}>{label}</div>
      <input type="number" value={value} min={min} max={max}
        onChange={(e) => onChange(Number(e.target.value))} style={{
          width: "100%", background: "#1c1c1e", border: "1px solid #2c2c2e",
          borderRadius: 6, color: "#f2f2f7", padding: "7px 10px",
          fontSize: 12, fontFamily: "monospace", boxSizing: "border-box",
        }} />
    </div>
  );
}

function MetCard({ label, value, sub, color = "#FF9F0A" }) {
  return (
    <div style={{
      background: "rgba(15,15,15,0.78)", backdropFilter: "blur(14px)",
      border: `1px solid ${color}28`, borderRadius: 10,
      padding: "14px 18px", flex: 1,
    }}>
      <div style={{ fontSize: 9, color: "#48484a", letterSpacing: "0.15em",
        textTransform: "uppercase", fontFamily: "monospace", marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 900, color,
        fontFamily: "'Syne', sans-serif", letterSpacing: "-0.02em" }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "#48484a", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SL({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
      <div style={{ width: 2, height: 13, background: "#FF9F0A", borderRadius: 1 }} />
      <span style={{ fontSize: 9, color: "#636366", letterSpacing: "0.18em",
        textTransform: "uppercase", fontFamily: "monospace" }}>{children}</span>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════════════════════════ */
export default function App() {
  const [form, setForm]       = useState(DEFAULTS);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [model, setModel]     = useState("rf");

  const set = (k) => (v) => setForm((f) => ({ ...f, [k]: v }));
  const tog = (k) => ()  => setForm((f) => ({ ...f, [k]: !f[k] }));

  const predict = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, model }),
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const risk  = result?.risk_level || "MEDIUM";
  const rCfg  = RISK[risk];
  const survData = result
    ? result.survival_curve.tenures.map((t, i) => ({
        t, survival: result.survival_curve.survival[i],
        hazard: result.survival_curve.hazard[i],
      }))
    : [];

  /* ─── RENDER ────────────────────────────────────────────────── */
  return (
    <div style={{ minHeight: "100vh", color: "#f2f2f7", position: "relative",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif" }}>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #000; }
        select, input[type=number] { outline: none !important; }
        select:focus, input[type=number]:focus { border-color: #FF9F0A !important; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #2c2c2e; border-radius: 2px; }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.3; }
      `}</style>

      {/* ── Background: dim the churn image ─────────────────────── */}
      <div style={{
        position: "fixed", inset: 0, zIndex: 0,
        backgroundImage: `url(${BG_IMAGE})`,
        backgroundSize: "cover", backgroundPosition: "center",
        filter: "brightness(0.14) saturate(0.3)",
      }} />
      {/* ── Gradient vignette over the bg ─────────────────────────── */}
      <div style={{
        position: "fixed", inset: 0, zIndex: 1,
        background: "linear-gradient(160deg, rgba(0,0,0,0.96) 0%, rgba(12,6,0,0.92) 100%)",
      }} />

      {/* ── All content ───────────────────────────────────────────── */}
      <div style={{ position: "relative", zIndex: 2 }}>

        {/* HEADER */}
        <header style={{
          borderBottom: "1px solid #1c1c1e",
          background: "rgba(0,0,0,0.55)", backdropFilter: "blur(20px)",
          padding: "18px 40px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{
              width: 38, height: 38, borderRadius: 9, flexShrink: 0,
              background: "linear-gradient(135deg, #FF9F0A, #FF3B30)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 18,
            }}>🚪</div>
            <div>
              <div style={{ fontSize: 21, fontWeight: 900, fontFamily: "'Syne', sans-serif",
                letterSpacing: "-0.025em" }}>Churn Intelligence</div>
              <div style={{ fontSize: 9, color: "#48484a", fontFamily: "monospace",
                letterSpacing: "0.12em", marginTop: 2 }}>
                TELECOM ANALYTICS · DECISION TREE &amp; RANDOM FOREST
              </div>
            </div>
          </div>

          {/* Model switcher */}
          <div style={{ display: "flex", background: "#111", borderRadius: 8,
            border: "1px solid #2c2c2e", overflow: "hidden" }}>
            {[{ id: "rf", lbl: "Random Forest ★" }, { id: "xgb", lbl: "XGBoost ⚡" }, { id: "dt", lbl: "Decision Tree" },].map((m) => (
              <button key={m.id} onClick={() => setModel(m.id)} style={{
                padding: "7px 18px", border: "none", cursor: "pointer",
                background: model === m.id ? "blue" : "transparent",
                color:      model === m.id ? "#000" : "#636366",
                fontFamily: "monospace", fontSize: 11,
                fontWeight: model === m.id ? 700 : 400, transition: "all 0.2s",
              }}>{m.lbl}</button>
            ))}
          </div>
        </header>

        {/* MAIN GRID */}
        <main style={{
          maxWidth: 1380, margin: "0 auto", padding: "26px 40px",
          display: "grid", gridTemplateColumns: "380px 1fr", gap: 22, alignItems: "start",
        }}>

          {/* ── FORM ──────────────────────────────────────────────── */}
          <div style={{
            background: "rgba(8,8,8,0.78)", backdropFilter: "blur(20px)",
            border: "1px solid #1c1c1e", borderRadius: 14,
            padding: "22px 18px", display: "flex", flexDirection: "column", gap: 22,
          }}>
            {/* Row 1 */}
            <div>
              <SL>Account Profile</SL>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {CHECKBOX_ROW1.map(({ key, label }) => (
                  <Chip key={key} checked={form[key]} onChange={tog(key)} label={label} />
                ))}
              </div>
            </div>

            {/* Row 2 */}
            <div>
              <SL>Services Subscribed</SL>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {CHECKBOX_ROW2.map(({ key, label }) => (
                  <Chip key={key} checked={form[key]} onChange={tog(key)} label={label} />
                ))}
              </div>
            </div>

            {/* Dropdowns */}
            <div>
              <SL>Customer Details</SL>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 14px" }}>
                <Sel label="Gender" value={form.gender} onChange={set("gender")}
                  options={["Male", "Female"]} />
                <Sel label="Internet Service" value={form.internetService}
                  onChange={set("internetService")} options={["DSL", "Fiber optic", "No"]} />
                <Sel label="Contract Type" value={form.contract} onChange={set("contract")}
                  options={["Month-to-Month", "One year", "Two year"]} />
                <Sel label="Payment Method" value={form.paymentMethod}
                  onChange={set("paymentMethod")}
                  options={["Electronic Check", "Mailed Check", "Bank Transfer (auto)", "Credit Card (auto)"]} />
              </div>
            </div>

            {/* Numerics */}
            <div>
              <SL>Usage Metrics</SL>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 14px" }}>
                <Num label="Tenure (months)" value={form.tenure}
                  onChange={set("tenure")} min={0} max={72} />
                <Num label="Monthly Charges ($)" value={form.monthlyCharges}
                  onChange={set("monthlyCharges")} min={0} max={200} />
              </div>
              <div style={{
                marginTop: 10, padding: "8px 12px", background: "#111",
                borderRadius: 6, fontSize: 10, color: "#48484a", fontFamily: "monospace",
              }}>
                Total Charges (computed):{" "}
                <span style={{ color: "#FF9F0A" }}>
                  ${(form.tenure * form.monthlyCharges).toLocaleString()}
                </span>
              </div>
            </div>

            {/* CTA */}
            <button onClick={predict} disabled={loading} style={{
              width: "100%", padding: "13px 0", borderRadius: 8, border: "none",
              background: loading ? "#1c1c1e" : "linear-gradient(135deg,#FF9F0A,#FF3B30)",
              color: loading ? "#48484a" : "#000",
              fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
              letterSpacing: "0.06em", fontFamily: "monospace", transition: "all 0.2s",
              boxShadow: loading ? "none" : "0 4px 22px #FF3B3045",
            }}>
              {loading ? "⟳  ANALYSING..." : "▶  RUN CHURN ANALYSIS"}
            </button>

            {error && (
              <div style={{ background: "#1c0000", border: "1px solid #FF3B3055",
                borderRadius: 6, padding: "10px 12px",
                fontSize: 11, color: "#FF453A", fontFamily: "monospace" }}>
                ✕ {error}
              </div>
            )}
          </div>

          {/* ── RESULTS ───────────────────────────────────────────── */}
          {!result ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center",
              justifyContent: "center", height: 500, gap: 14 }}>
              <div style={{ fontSize: 64, opacity: 0.15 }}>🚪</div>
              <div style={{ fontSize: 11, fontFamily: "monospace", color: "#3a3a3c",
                letterSpacing: "0.14em" }}>SET INPUTS · CLICK ANALYSE</div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

              {/* Metric strip */}
              <div style={{ display: "flex", gap: 14 }}>
                <MetCard label="Churn Probability"
                  value={`${(result.churn_probability * 100).toFixed(1)}%`}
                  sub={`Risk Level: ${result.risk_level}`} color={rCfg.color} />
                <MetCard label="Expected LTV"
                  value={`$${result.ltv.toLocaleString()}`}
                  sub="Projected revenue before churn" color="#FF9F0A" />
                <MetCard label="Model Engine"
                  value={result.model_used || model.toUpperCase()}
                  sub="Active prediction backend" color="#FFD60A" />
              </div>

              {/* Gauge + Line charts */}
              <div style={{ display: "grid", gridTemplateColumns: "195px 1fr 1fr", gap: 16 }}>
                {/* Gauge */}
                <div style={{
                  background: "rgba(8,8,8,0.78)", backdropFilter: "blur(16px)",
                  border: `1px solid ${rCfg.color}33`, borderRadius: 12,
                  padding: "18px 10px",
                  display: "flex", flexDirection: "column", alignItems: "center",
                }}>
                  <SL>Risk Gauge</SL>
                  <Gauge prob={result.churn_probability} risk={risk} />
                </div>

                {/* Survival */}
                <div style={{ background: "rgba(8,8,8,0.78)", backdropFilter: "blur(16px)",
                  border: "1px solid #1c1c1e", borderRadius: 12, padding: "18px" }}>
                  <SL>Survival Probability</SL>
                  <ResponsiveContainer width="100%" height={165}>
                    <AreaChart data={survData} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
                      <defs>
                        <linearGradient id="survG" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#FF9F0A" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#FF9F0A" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="2 4" stroke="#1c1c1e" />
                      <XAxis dataKey="t" tick={{ fill: "#3a3a3c", fontSize: 9 }} />
                      <YAxis tick={{ fill: "#3a3a3c", fontSize: 9 }} domain={[0, 1]} />
                      <Tooltip content={<CT />} />
                      <ReferenceLine x={result.survival_curve.current_tenure}
                        stroke="#FF9F0A55" strokeDasharray="3 3"
                        label={{ value: "now", fill: "#FF9F0A", fontSize: 8, position: "top" }} />
                      <Area type="monotone" dataKey="survival"
                        stroke="#FF9F0A" strokeWidth={2} fill="url(#survG)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Hazard */}
                <div style={{ background: "rgba(8,8,8,0.78)", backdropFilter: "blur(16px)",
                  border: "1px solid #1c1c1e", borderRadius: 12, padding: "18px" }}>
                  <SL>Cumulative Hazard</SL>
                  <ResponsiveContainer width="100%" height={165}>
                    <LineChart data={survData} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke="#1c1c1e" />
                      <XAxis dataKey="t" tick={{ fill: "#3a3a3c", fontSize: 9 }} />
                      <YAxis tick={{ fill: "#3a3a3c", fontSize: 9 }} />
                      <Tooltip content={<CT />} />
                      <ReferenceLine x={result.survival_curve.current_tenure}
                        stroke="#FF3B3055" strokeDasharray="3 3"
                        label={{ value: "now", fill: "#FF3B30", fontSize: 8, position: "top" }} />
                      <Line type="monotone" dataKey="hazard"
                        stroke="#FF3B30" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* SHAP */}
              <div style={{ background: "rgba(8,8,8,0.78)", backdropFilter: "blur(16px)",
                border: "1px solid #1c1c1e", borderRadius: 12, padding: "22px 24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between",
                  alignItems: "center", marginBottom: 18 }}>
                  <SL>SHAP Feature Impact</SL>
                  <div style={{ fontSize: 9, color: "#3a3a3c", fontFamily: "monospace" }}>
                    output → {result.churn_probability.toFixed(4)}
                  </div>
                </div>
                <ShapBars data={result.shap_values} />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}