import { useState, useRef } from "react";

const fmt = (n, d = 2) => n == null ? "—" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtPct = (n) => n == null ? "—" : `${n >= 0 ? "+" : ""}${Number(n).toFixed(2)}%`;
const fmtLarge = (n) => {
  if (!n) return "—";
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  return `$${fmt(n)}`;
};

const COIN_COLORS = { BTC: "#F7931A", ETH: "#627EEA", SOL: "#9945FF", BNB: "#F3BA2F", XRP: "#00AAE4", DOGE: "#C2A633" };
const COOLDOWN_SECONDS = 30;
const GEMINI_MODEL = "gemini-2.0-flash";

const DATA_PROMPT = `Search the web for CURRENT live cryptocurrency prices right now and return ONLY a JSON object. No explanation, no markdown, just raw JSON.

Search for: BTC price, ETH price, SOL price, BNB price, XRP price, DOGE price, crypto fear greed index, total crypto market cap

Return this exact structure:
{
  "timestamp": "current UTC time",
  "fearGreed": { "value": 65, "label": "Greed" },
  "totalMarketCap": 3200000000000,
  "btcDominance": 54.2,
  "coins": [
    { "symbol": "BTC", "price": 104000, "change24h": 1.5, "marketCap": 2050000000000, "volume24h": 38000000000 },
    { "symbol": "ETH", "price": 2500, "change24h": -0.8, "marketCap": 300000000000, "volume24h": 15000000000 },
    { "symbol": "SOL", "price": 165, "change24h": 2.1, "marketCap": 78000000000, "volume24h": 4000000000 },
    { "symbol": "BNB", "price": 620, "change24h": 0.5, "marketCap": 90000000000, "volume24h": 2000000000 },
    { "symbol": "XRP", "price": 2.3, "change24h": -1.2, "marketCap": 130000000000, "volume24h": 8000000000 },
    { "symbol": "DOGE", "price": 0.18, "change24h": 3.2, "marketCap": 26000000000, "volume24h": 2000000000 }
  ]
}

Use real searched values. Return ONLY the JSON, nothing else.`;

const ANALYSIS_PROMPT = (data, mode) => `You are an AI Crypto Market Intelligence Agent. Analyze this LIVE market data and generate a ${mode === "daily" ? "complete daily intelligence report" : "priority alert assessment"}.

LIVE DATA:
${JSON.stringify(data, null, 2)}

${mode === "daily" ? `Use these exact section headers:
## MARKET OVERVIEW
## BTC ANALYSIS
## ETH ANALYSIS
## SECTOR PERFORMANCE
## RISK ASSESSMENT
## AI MARKET OUTLOOK` : "Focus on CRITICAL/HIGH alerts only. Use: ⚠️ ALERT [CRITICAL/HIGH/MEDIUM/LOW]: Title"}

Rules: hedge fund analyst tone, no hype, under 500 words, actionable intelligence only.`;

// Call Gemini API with Google Search grounding tool
async function callGemini(apiKey, prompt, stream = false) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:${stream ? "streamGenerateContent" : "generateContent"}?key=${apiKey}${stream ? "&alt=sse" : ""}`;

  const body = {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    tools: [{ google_search: {} }],
  };

  let delay = 5000;
  for (let attempt = 0; attempt <= 3; attempt++) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 429) {
      if (attempt === 3) throw new Error("Rate limited (429). Please wait and try again.");
      await new Promise(r => setTimeout(r, delay));
      delay *= 2;
      continue;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Gemini API error ${res.status}`);
    }
    return res;
  }
}

// Extract text from Gemini non-streaming response
function extractGeminiText(json) {
  return (json.candidates?.[0]?.content?.parts || [])
    .map(p => p.text || "")
    .join("");
}

function PriceCard({ coin }) {
  const up = coin.change24h >= 0;
  const color = COIN_COLORS[coin.symbol] || "#888";
  return (
    <div style={{ border: `1px solid ${up ? "#0a2a1a" : "#2a0a0a"}`, background: up ? "#030d07" : "#0d0303", borderRadius: "4px", padding: "11px 13px", fontFamily: "'IBM Plex Mono',monospace", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: color }} />
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span style={{ color, fontSize: "11px", fontWeight: 700 }}>{coin.symbol}</span>
        <span style={{ color: up ? "#00ff88" : "#ff4444", fontSize: "10px", fontWeight: 700, background: up ? "#001a0d" : "#1a0000", padding: "1px 5px", borderRadius: "3px" }}>{fmtPct(coin.change24h)}</span>
      </div>
      <div style={{ color: "#e8ffe8", fontSize: "16px", fontWeight: 700, margin: "5px 0 3px" }}>${fmt(coin.price)}</div>
      <div style={{ color: "#1a4a1a", fontSize: "9px" }}>MCap {fmtLarge(coin.marketCap)}</div>
      <div style={{ color: "#1a3a1a", fontSize: "9px", marginTop: "2px" }}>Vol {fmtLarge(coin.volume24h)}</div>
    </div>
  );
}

function FearGreed({ value, label }) {
  if (!value) return null;
  const v = parseInt(value);
  const color = v <= 25 ? "#ff4444" : v <= 45 ? "#ff8844" : v <= 55 ? "#ffcc44" : v <= 75 ? "#88ee44" : "#00ff88";
  const pct = v / 100;
  const angle = -90 + pct * 180;
  return (
    <div style={{ textAlign: "center" }}>
      <svg width="100" height="58" viewBox="0 0 120 70">
        <path d="M10 65 A50 50 0 0 1 110 65" fill="none" stroke="#1a2a1a" strokeWidth="12" />
        <path d="M10 65 A50 50 0 0 1 110 65" fill="none" stroke={color} strokeWidth="12" strokeDasharray={`${pct * 157} 157`} opacity="0.9" />
        <line x1="60" y1="65" x2={60 + 36 * Math.cos(angle * Math.PI / 180)} y2={65 + 36 * Math.sin(angle * Math.PI / 180)} stroke={color} strokeWidth="2" />
        <circle cx="60" cy="65" r="4" fill={color} />
        <text x="60" y="50" textAnchor="middle" fill={color} fontSize="17" fontWeight="700" fontFamily="monospace">{v}</text>
      </svg>
      <div style={{ color, fontSize: "9px", fontWeight: 700, letterSpacing: "1px", marginTop: "-4px" }}>{(label || "").toUpperCase()}</div>
    </div>
  );
}

function ReportView({ text, loading }) {
  if (!text && !loading) return null;
  return (
    <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: "12px", lineHeight: "1.8", color: "#b0d4b0" }}>
      {(text || "").split("\n").map((line, i) => {
        if (line.startsWith("## ")) return <div key={i} style={{ color: "#00ff88", fontWeight: 700, fontSize: "9px", letterSpacing: "2px", margin: "14px 0 5px", borderLeft: "2px solid #00ff88", paddingLeft: "8px" }}>{line.replace("## ", "")}</div>;
        if (line.startsWith("⚠️ ALERT")) {
          const c = line.includes("CRITICAL") ? "#ff4444" : line.includes("HIGH") ? "#ffaa44" : "#ffdd44";
          return <div key={i} style={{ background: "#0f0800", border: `1px solid ${c}`, borderRadius: "4px", padding: "7px 11px", margin: "7px 0", color: c, fontSize: "11px", fontWeight: 700 }}>{line}</div>;
        }
        if (line.startsWith("- ")) return <div key={i} style={{ paddingLeft: "10px", color: "#88b888" }}><span style={{ color: "#00aa55" }}>›</span> {line.slice(2)}</div>;
        if (!line.trim()) return <div key={i} style={{ height: "5px" }} />;
        return <div key={i}>{line}</div>;
      })}
      {loading && <span style={{ animation: "blink 1s infinite", color: "#00ff88" }}>▌</span>}
    </div>
  );
}

export default function App() {
  const [apiKey, setApiKey] = useState("");
  const [keyConfirmed, setKeyConfirmed] = useState(false);
  const [coins, setCoins] = useState([]);
  const [meta, setMeta] = useState(null);
  const [report, setReport] = useState("");
  const [streamText, setStreamText] = useState("");
  const [phase, setPhase] = useState("idle");
  const [error, setError] = useState("");
  const [mode, setMode] = useState("daily");
  const [synced, setSynced] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [cooldown, setCooldown] = useState(0);
  const cooldownRef = useRef(null);

  const startCooldown = () => {
    setCooldown(COOLDOWN_SECONDS);
    clearInterval(cooldownRef.current);
    cooldownRef.current = setInterval(() => {
      setCooldown(prev => {
        if (prev <= 1) { clearInterval(cooldownRef.current); return 0; }
        return prev - 1;
      });
    }, 1000);
  };

  const run = async () => {
    if (cooldown > 0 || !keyConfirmed) return;
    setPhase("fetching"); setError(""); setReport(""); setStreamText(""); setCoins([]); setMeta(null);
    setStatusMsg("Searching web for live prices via Gemini...");
    try {
      // Step 1: fetch live data with Google Search grounding
      const dataRes = await callGemini(apiKey, DATA_PROMPT, false);
      const dataJson = await dataRes.json();
      const rawText = extractGeminiText(dataJson).trim();
      let marketData;
      try {
        const clean = rawText.replace(/```json|```/g, "").trim();
        marketData = JSON.parse(clean);
      } catch {
        throw new Error("Could not parse market data from Gemini. Try again.");
      }

      setCoins(marketData.coins || []);
      setMeta({ fearGreed: marketData.fearGreed, totalMarketCap: marketData.totalMarketCap, btcDominance: marketData.btcDominance, timestamp: marketData.timestamp });
      setSynced(new Date());

      // Step 2: stream analysis
      setPhase("analyzing");
      setStatusMsg("Gemini generating intelligence report...");
      const anaRes = await callGemini(apiKey, ANALYSIS_PROMPT(marketData, mode), true);
      const reader = anaRes.body.getReader();
      const dec = new TextDecoder();
      let full = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) {
            try {
              const j = JSON.parse(line.slice(6));
              const text = (j.candidates?.[0]?.content?.parts || []).map(p => p.text || "").join("");
              if (text) { full += text; setStreamText(full); }
            } catch {}
          }
        }
      }
      setReport(full); setStreamText(""); setPhase("done");
      startCooldown();
    } catch (e) {
      setError(e.message); setPhase("error");
      if (e.message.includes("429") || e.message.toLowerCase().includes("rate limit")) {
        startCooldown();
      }
    }
  };

  const display = streamText || report;
  const btc = coins.find(c => c.symbol === "BTC");
  const isRunning = phase === "fetching" || phase === "analyzing";
  const canRun = !isRunning && cooldown === 0 && keyConfirmed;

  // API Key entry screen
  if (!keyConfirmed) {
    return (
      <div style={{ minHeight: "100vh", background: "#020702", color: "#c0dcc0", fontFamily: "'IBM Plex Mono',monospace", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Orbitron:wght@900&display=swap');button{cursor:pointer;font-family:'IBM Plex Mono',monospace;transition:all .15s}button:hover{filter:brightness(1.2)}`}</style>
        <div style={{ border: "1px solid #0a3a1a", borderRadius: "6px", padding: "30px 32px", background: "#030e06", maxWidth: "400px", width: "90%" }}>
          <div style={{ fontFamily: "'Orbitron'", fontSize: "13px", fontWeight: 900, color: "#00ff88", letterSpacing: "3px", marginBottom: "6px" }}>◈ CRYPTO INTEL</div>
          <div style={{ fontSize: "8px", color: "#2a5a2a", letterSpacing: "2px", marginBottom: "22px" }}>POWERED BY GEMINI · GOOGLE SEARCH</div>
          <div style={{ color: "#3a8a4a", fontSize: "10px", marginBottom: "8px", letterSpacing: "1px" }}>▸ ENTER GEMINI API KEY</div>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            onKeyDown={e => e.key === "Enter" && apiKey.trim() && setKeyConfirmed(true)}
            placeholder="AIza..."
            style={{ width: "100%", background: "#010801", border: "1px solid #0a3a1a", borderRadius: "3px", padding: "9px 11px", color: "#00ff88", fontSize: "11px", fontFamily: "'IBM Plex Mono',monospace", outline: "none", boxSizing: "border-box", marginBottom: "10px" }}
          />
          <button
            onClick={() => apiKey.trim() && setKeyConfirmed(true)}
            disabled={!apiKey.trim()}
            style={{ width: "100%", background: "#002816", border: "1px solid #00aa55", color: "#00ff88", padding: "9px", borderRadius: "3px", fontSize: "11px", fontWeight: 700 }}
          >
            ◈ CONNECT
          </button>
          <div style={{ marginTop: "14px", color: "#1a3a1a", fontSize: "9px", lineHeight: "1.7" }}>
            Your key is stored only in memory and never saved.<br />
            Get a free key at <span style={{ color: "#2a6a2a" }}>aistudio.google.com</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", background: "#020702", color: "#c0dcc0", fontFamily: "'IBM Plex Mono',monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Orbitron:wght@900&display=swap');
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
        @keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}
        @keyframes fin{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
        button{cursor:pointer;transition:all .15s;font-family:'IBM Plex Mono',monospace}
        button:hover:not(:disabled){filter:brightness(1.2);transform:translateY(-1px)}
        button:disabled{opacity:.35;cursor:not-allowed}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:#1a4a1a}
        input:focus{border-color:#00aa55 !important}
      `}</style>

      {/* Header */}
      <div style={{ background: "#020d02", borderBottom: "1px solid #0a2a0a", padding: "10px 18px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontFamily: "'Orbitron'", fontSize: "14px", fontWeight: 900, color: "#00ff88", letterSpacing: "3px" }}>◈ CRYPTO INTEL</div>
          <div style={{ fontSize: "8px", color: "#2a5a2a", letterSpacing: "2px" }}>AI MARKET INTELLIGENCE · GEMINI + GOOGLE SEARCH</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <button onClick={() => { setKeyConfirmed(false); setApiKey(""); setCoins([]); setMeta(null); setReport(""); setPhase("idle"); }} style={{ background: "none", border: "1px solid #0a3a1a", color: "#2a5a2a", padding: "3px 8px", borderRadius: "3px", fontSize: "9px" }}>⏏ KEY</button>
          {synced && <span style={{ fontSize: "9px", color: "#2a5a2a" }}>{synced.toLocaleTimeString()}</span>}
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: isRunning ? "#ffcc00" : coins.length ? "#00ff88" : "#334433", animation: "pulse 2s infinite" }} />
        </div>
      </div>

      {/* Live ticker */}
      {coins.length > 0 && (
        <div style={{ overflow: "hidden", background: "#050d05", borderBottom: "1px solid #0a1a0a", padding: "5px 0" }}>
          <div style={{ display: "flex", gap: "32px", whiteSpace: "nowrap", animation: "tick 20s linear infinite", fontSize: "10px" }}>
            {[...coins, ...coins].map((c, i) => (
              <span key={i} style={{ display: "inline-flex", gap: "7px" }}>
                <span style={{ color: COIN_COLORS[c.symbol] || "#aaa", fontWeight: 700 }}>{c.symbol}</span>
                <span style={{ color: "#ddd" }}>${fmt(c.price)}</span>
                <span style={{ color: c.change24h >= 0 ? "#00ff88" : "#ff4444" }}>{fmtPct(c.change24h)}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ padding: "14px 18px", maxWidth: "1080px", margin: "0 auto" }}>

        {phase === "error" && (
          <div style={{ background: "#1a0505", border: "1px solid #550000", borderRadius: "4px", padding: "9px 13px", marginBottom: "10px", color: "#ff7766", fontSize: "11px" }}>
            ✗ {error}
            {(error.includes("429") || error.toLowerCase().includes("rate limit")) && (
              <div style={{ marginTop: "5px", color: "#aa4433", fontSize: "10px" }}>Rate limited — wait for cooldown then retry.</div>
            )}
            {error.includes("API_KEY") || error.toLowerCase().includes("api key") ? (
              <div style={{ marginTop: "5px", color: "#aa4433", fontSize: "10px" }}>Invalid API key. Click ⏏ KEY to re-enter.</div>
            ) : null}
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "10px" }}>
          <div style={{ border: "1px solid #0a2a1a", borderRadius: "4px", padding: "12px 14px", background: "#030e06" }}>
            <div style={{ color: "#00cc66", fontSize: "9px", letterSpacing: "2px", fontWeight: 700, marginBottom: "8px", borderBottom: "1px solid #0a2a1a", paddingBottom: "4px" }}>▸ AGENT CONTROLS</div>
            <div style={{ display: "flex", gap: "6px", marginBottom: "10px" }}>
              {[["daily","📊 DAILY"],["alert","⚠️ ALERTS"]].map(([m,l]) => (
                <button key={m} onClick={() => setMode(m)} disabled={isRunning} style={{ background: mode===m?"#003a1a":"#050d05", border:`1px solid ${mode===m?"#00aa55":"#1a3a1a"}`, color:mode===m?"#00ff88":"#3a7a3a", padding:"4px 10px", borderRadius:"3px", fontSize:"10px", fontWeight:700, letterSpacing:"1px" }}>{l}</button>
              ))}
            </div>
            <button onClick={run} disabled={!canRun} style={{ background: !canRun?"#001a0d":"#002816", border:`1px solid ${!canRun?"#004422":"#00aa55"}`, color:!canRun?"#2a5a2a":"#00ff88", padding:"9px 20px", borderRadius:"3px", fontSize:"11px", fontWeight:700, width:"100%" }}>
              {isRunning ? "◈ RUNNING..." : cooldown > 0 ? `⏳ COOLDOWN ${cooldown}s` : "◈ FETCH DATA & ANALYZE"}
            </button>
            {isRunning && (
              <div style={{ marginTop: "10px", color: "#3a8a4a", fontSize: "10px", display: "flex", gap: "6px", alignItems: "center" }}>
                <span style={{ animation: "blink 1s infinite" }}>▌</span>
                <span>{statusMsg}</span>
              </div>
            )}
            {cooldown > 0 && !isRunning && (
              <div style={{ marginTop: "8px", color: "#4a6a2a", fontSize: "9px" }}>Next run in {cooldown}s</div>
            )}
            <div style={{ marginTop: "10px", color: "#1a4a1a", fontSize: "9px", lineHeight: "1.5" }}>
              Uses Gemini + Google Search grounding for live prices.
            </div>
          </div>

          <div style={{ border: "1px solid #0a2a1a", borderRadius: "4px", padding: "12px 14px", background: "#030e06" }}>
            <div style={{ color: "#00cc66", fontSize: "9px", letterSpacing: "2px", fontWeight: 700, marginBottom: "8px", borderBottom: "1px solid #0a2a1a", paddingBottom: "4px" }}>▸ MARKET PULSE</div>
            {meta ? (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "5px", marginBottom: "8px" }}>
                  {[
                    ["TOTAL MCAP", fmtLarge(meta.totalMarketCap)],
                    ["BTC PRICE", btc ? `$${fmt(btc.price)}` : "—"],
                    ["BTC DOM", meta.btcDominance ? `${meta.btcDominance}%` : "—"],
                    ["BTC 24H", btc ? fmtPct(btc.change24h) : "—"],
                  ].map(([k,v]) => (
                    <div key={k} style={{ background:"#020902", border:"1px solid #0a1a0a", borderRadius:"3px", padding:"5px 7px" }}>
                      <div style={{ color:"#1a4a1a", fontSize:"8px", letterSpacing:"1px" }}>{k}</div>
                      <div style={{ color: k==="BTC 24H"?(btc?.change24h>=0?"#00ff88":"#ff4444"):"#a0e0a0", fontSize:"11px", fontWeight:700, marginTop:"2px" }}>{v}</div>
                    </div>
                  ))}
                </div>
                <FearGreed value={meta.fearGreed?.value} label={meta.fearGreed?.label} />
              </>
            ) : (
              <div style={{ color: "#1a4a1a", fontSize: "11px", padding: "20px", textAlign: "center" }}>
                {isRunning ? <span style={{ animation: "blink 1s infinite" }}>Fetching live data...</span> : "Run analysis to load data"}
              </div>
            )}
          </div>
        </div>

        {coins.length > 0 && (
          <div style={{ marginBottom: "12px", animation: "fin .4s ease" }}>
            <div style={{ color:"#00cc66", fontSize:"9px", letterSpacing:"2px", fontWeight:700, marginBottom:"7px", borderBottom:"1px solid #0a2a1a", paddingBottom:"4px" }}>▸ LIVE PRICES · GOOGLE SEARCH</div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(155px,1fr))", gap:"7px" }}>
              {coins.map(c => <PriceCard key={c.symbol} coin={c} />)}
            </div>
          </div>
        )}

        {!coins.length && !isRunning && (
          <div style={{ textAlign:"center", padding:"50px 20px", color:"#1a4a1a", fontSize:"12px", lineHeight:"2" }}>
            <div style={{ fontSize:"28px", marginBottom:"8px" }}>◈</div>
            Hit <strong style={{color:"#00cc66"}}>◈ FETCH DATA & ANALYZE</strong> to start<br/>
            <span style={{ fontSize:"10px", color:"#0f3a0f" }}>Gemini will search Google for live prices then generate your report</span>
          </div>
        )}

        {(display || (isRunning && phase === "analyzing")) && (
          <div style={{ border:"1px solid #0a3a1a", borderRadius:"4px", padding:"15px 17px", background:"#020d04", animation:"fin .3s ease" }}>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:"11px" }}>
              <div style={{ fontFamily:"'Orbitron'", fontSize:"9px", color:"#00ff88", letterSpacing:"3px", fontWeight:900 }}>
                ◈ {mode==="daily" ? "DAILY INTELLIGENCE REPORT" : "ALERT ASSESSMENT"}
              </div>
              {synced && <div style={{ fontSize:"8px", color:"#2a5a2a" }}>{synced.toUTCString()}</div>}
            </div>
            <ReportView text={display} loading={phase === "analyzing"} />
          </div>
        )}

        <div style={{ marginTop:"14px", textAlign:"center", color:"#0f2a0f", fontSize:"8px", letterSpacing:"1px" }}>
          CRYPTO INTEL · INFORMATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE
        </div>
      </div>
    </div>
  );
}
