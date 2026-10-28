import React, { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, Legend,
} from 'recharts';
import { apiService } from '../services/api';
import { ArrowLeft, ChevronDown, ChevronUp, Clock, BarChart2, GitBranch, Zap } from 'lucide-react';

/* ─── Styles ─────────────────────────────────────────────────────────────────── */
const Styles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&family=Cinzel+Decorative:wght@700&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0a0f1a; --glass-bg: rgba(20,35,55,0.55);
      --glass-border: rgba(201,168,76,0.18); --gold: #c9a84c;
      --text: rgba(248,235,190,0.82); --text-dim: rgba(220,195,130,0.5);
      --font-sans: 'DM Sans', sans-serif; --font-serif: 'DM Serif Display', serif;
      --font-brand: 'Cinzel Decorative', cursive;
    }
    .pl-wrap { font-family: var(--font-sans); background: var(--bg); min-height: 100vh; color: #fdf8ee; }
    .pl-nav {
      position: sticky; top: 0; z-index: 100; height: 60px;
      display: flex; align-items: center; padding: 0 clamp(20px,5vw,72px);
      justify-content: space-between;
      background: rgba(10,15,26,0.8); backdrop-filter: blur(28px);
      border-bottom: 1px solid var(--glass-border);
    }
    .pl-nav-brand { font-family: var(--font-brand); font-size: 1.15rem; color: var(--gold); text-decoration: none; }
    .pl-back { display: flex; align-items: center; gap: 6px; color: var(--text); text-decoration: none; font-size: 0.82rem; transition: color .2s; }
    .pl-back:hover { color: var(--gold); }
    .pl-page { max-width: 1080px; margin: 0 auto; padding: clamp(32px,5vw,64px) clamp(20px,5vw,48px) 80px; }
    .pl-title { font-family: var(--font-serif); font-size: 2rem; color: #fdf8ee; margin-bottom: 6px; }
    .pl-subtitle { font-size: 0.82rem; color: var(--text-dim); margin-bottom: 40px; }

    /* Performance bar */
    .pl-perf-bar { display: flex; border-radius: 12px; overflow: hidden; height: 32px; margin-bottom: 40px; border: 1px solid var(--glass-border); }
    .pl-perf-seg { display: flex; align-items: center; justify-content: center; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.06em; color: #0a0f1a; transition: flex .5s ease; white-space: nowrap; overflow: hidden; padding: 0 4px; }
    .pl-perf-legend { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 28px; }
    .pl-perf-legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.74rem; color: var(--text-dim); }
    .pl-perf-swatch { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }

    /* Panels */
    .pl-panel { border: 1px solid var(--glass-border); border-radius: 20px; overflow: hidden; margin-bottom: 20px; background: var(--glass-bg); backdrop-filter: blur(24px); }
    .pl-panel-header {
      display: flex; align-items: center; gap: 14px; padding: 18px 24px;
      cursor: pointer; user-select: none;
      border-bottom: 1px solid transparent; transition: border-color .2s;
    }
    .pl-panel-header.open { border-bottom-color: var(--glass-border); }
    .pl-panel-icon { width: 38px; height: 38px; border-radius: 12px; background: rgba(201,168,76,0.1); border: 1px solid rgba(201,168,76,0.22); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .pl-panel-label { flex: 1; }
    .pl-panel-title { font-family: var(--font-serif); font-size: 1.05rem; color: #fdf8ee; }
    .pl-panel-desc { font-size: 0.75rem; color: var(--text-dim); margin-top: 2px; }
    .pl-panel-badge { font-size: 0.66rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; padding: 3px 10px; border-radius: 100px; background: rgba(201,168,76,0.1); color: var(--gold); border: 1px solid rgba(201,168,76,0.22); white-space: nowrap; }
    .pl-panel-body { padding: 24px; }

    /* Stats row */
    .pl-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .pl-stat { background: rgba(5,10,20,0.55); border: 1px solid rgba(201,168,76,0.08); border-radius: 12px; padding: 14px 16px; }
    .pl-stat-val { font-size: 1.6rem; font-weight: 600; color: var(--gold); font-family: 'Menlo', monospace; line-height: 1; }
    .pl-stat-label { font-size: 0.66rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px; }

    /* Chart wrapper */
    .pl-chart-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(201,168,76,0.55); margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
    .pl-chart-label::before { content: ''; width: 4px; height: 4px; border-radius: 50%; background: rgba(201,168,76,0.55); }
    .pl-chart-wrap { background: rgba(5,10,20,0.45); border: 1px solid rgba(201,168,76,0.08); border-radius: 14px; padding: 20px; margin-bottom: 20px; }

    /* KG graph */
    .pl-kg-svg { width: 100%; border-radius: 10px; background: rgba(5,10,20,0.65); overflow: visible; }

    /* Chunk list */
    .pl-chunk-list { display: flex; flex-direction: column; gap: 6px; max-height: 320px; overflow-y: auto; }
    .pl-chunk-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 10px; background: rgba(5,10,20,0.45); border: 1px solid rgba(201,168,76,0.07); }
    .pl-chunk-ts { font-size: 0.66rem; font-family: 'Menlo', monospace; color: var(--gold); background: rgba(201,168,76,0.1); border: 1px solid rgba(201,168,76,0.2); padding: 2px 7px; border-radius: 100px; white-space: nowrap; flex-shrink: 0; }
    .pl-chunk-score { font-size: 0.66rem; color: var(--text-dim); padding: 2px 7px; border: 1px solid rgba(201,168,76,0.08); border-radius: 100px; white-space: nowrap; flex-shrink: 0; }
    .pl-chunk-text { font-size: 0.78rem; color: var(--text); line-height: 1.55; }

    /* Keyframe thumb */
    .pl-kf-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; }
    .pl-kf-cell { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid rgba(201,168,76,0.18); aspect-ratio: 16/9; background: #000; }
    .pl-kf-cell img { width: 100%; height: 100%; object-fit: cover; }
    .pl-kf-ts { position: absolute; bottom: 4px; left: 4px; font-size: 0.6rem; font-family: 'Menlo', monospace; color: var(--gold); background: rgba(10,15,26,0.85); padding: 1px 5px; border-radius: 100px; }

    /* Loading / empty */
    .pl-loading { display: flex; align-items: center; justify-content: center; min-height: 200px; color: var(--text-dim); font-size: 0.88rem; }
    .pl-empty { padding: 32px 16px; text-align: center; color: var(--text-dim); font-size: 0.82rem; font-style: italic; }
  `}</style>
);

/* ─── Colour palette for clusters ─────────────────────────────────────────────── */
const CLUSTER_COLORS = [
  '#c9a84c','#5ba4cf','#e06c75','#98c379','#d19a66',
  '#61afef','#c678dd','#56b6c2','#f7b731','#a29bfe',
];
const clusterColor = (id) => CLUSTER_COLORS[id % CLUSTER_COLORS.length];

/* ─── Simple circular KG visualiser ──────────────────────────────────────────── */
const KGGraph = ({ nodes, edges }) => {
  const [hover, setHover] = useState(null);
  if (!nodes || nodes.length === 0) return <div className="pl-empty">No knowledge graph data.</div>;

  const W = 600, H = 400, CX = W / 2, CY = H / 2;
  const R = Math.min(CX, CY) - 55;
  const MAX_NODES = Math.min(nodes.length, 80);
  const displayed = nodes.slice(0, MAX_NODES);
  const nodeMap = {};
  displayed.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / displayed.length - Math.PI / 2;
    nodeMap[n.id] = { ...n, x: CX + R * Math.cos(angle), y: CY + R * Math.sin(angle) };
  });

  const displayedIds = new Set(displayed.map(n => n.id));
  const displayedEdges = (edges || []).filter(
    e => displayedIds.has(e.source) && displayedIds.has(e.target)
  ).slice(0, 120);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="pl-kg-svg" style={{ height: 380 }}>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5"
          markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(201,168,76,0.35)" />
        </marker>
      </defs>
      {displayedEdges.map((e, i) => {
        const src = nodeMap[e.source];
        const tgt = nodeMap[e.target];
        if (!src || !tgt) return null;
        const isHover = hover === e.source || hover === e.target;
        return (
          <line key={i} x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
            stroke={isHover ? 'rgba(201,168,76,0.55)' : 'rgba(201,168,76,0.12)'}
            strokeWidth={isHover ? 1.5 : 0.8}
            markerEnd="url(#arrow)"
          />
        );
      })}
      {Object.values(nodeMap).map((n) => {
        const isHover = hover === n.id;
        return (
          <g key={n.id} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}>
            <circle cx={n.x} cy={n.y} r={isHover ? 7 : 5}
              fill={isHover ? '#c9a84c' : 'rgba(201,168,76,0.45)'}
              stroke={isHover ? '#fdf8ee' : 'rgba(201,168,76,0.7)'}
              strokeWidth={isHover ? 1.5 : 1}
            />
            {isHover && (
              <text x={n.x} y={n.y - 12} textAnchor="middle"
                fill="#fdf8ee" fontSize={9} fontFamily="DM Sans, sans-serif">
                {n.id.length > 20 ? n.id.slice(0, 20) + '…' : n.id}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

/* ─── Collapsible panel ───────────────────────────────────────────────────────── */
const Panel = ({ icon: Icon, title, desc, badge, children, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="pl-panel">
      <div className={`pl-panel-header${open ? ' open' : ''}`} onClick={() => setOpen(o => !o)}>
        <div className="pl-panel-icon"><Icon size={17} color="#c9a84c" /></div>
        <div className="pl-panel-label">
          <div className="pl-panel-title">{title}</div>
          {desc && <div className="pl-panel-desc">{desc}</div>}
        </div>
        {badge && <div className="pl-panel-badge">{badge}</div>}
        {open ? <ChevronUp size={16} color="rgba(201,168,76,0.55)" /> : <ChevronDown size={16} color="rgba(201,168,76,0.55)" />}
      </div>
      {open && <div className="pl-panel-body">{children}</div>}
    </div>
  );
};

/* ─── Stat card ───────────────────────────────────────────────────────────────── */
const Stat = ({ val, label }) => (
  <div className="pl-stat">
    <div className="pl-stat-val">{val ?? '—'}</div>
    <div className="pl-stat-label">{label}</div>
  </div>
);

/* ─── Offered-vs-cited keyframes for one summary ───────────────────────────── */
const fmtClock = (sec) => `${Math.floor((sec || 0) / 60)}:${String(Math.floor((sec || 0) % 60)).padStart(2, '0')}`;

const RefBlock = ({ title, data }) => {
  if (!data) return null;
  return (
    <div style={{ marginBottom: 18 }}>
      <div className="pl-chart-label">{title}</div>
      <div className="pl-stats">
        <Stat val={data.offered_count} label="Keyframes offered" />
        <Stat val={data.cited_count} label="Cited by LLM" />
        <Stat val={`${Math.round((data.cited_ratio || 0) * 100)}%`} label="Cited / offered" />
        <Stat val={data.invalid?.length ?? 0} label="Invalid refs" />
      </div>
      {data.offered_count === 0 ? (
        <div style={{ fontSize: '0.8rem', color: 'rgba(220,195,130,0.5)', fontStyle: 'italic' }}>
          No keyframes were offered to this summary.
        </div>
      ) : (
        <div className="pl-chunk-list">
          {data.offered.map((a) => (
            <div key={a.keyframe_number} className="pl-chunk-item">
              <span className="pl-chunk-ts">KF {a.keyframe_number}</span>
              <span className="pl-chunk-score" style={{ color: a.cited ? '#98c379' : 'rgba(220,195,130,0.4)' }}>
                {a.cited ? 'cited' : 'not cited'}
              </span>
              <span className="pl-chunk-text">
                @{fmtClock(a.timestamp)} · chunk {fmtClock(a.time_start)}–{fmtClock(a.time_end)}
                {a.has_ocr ? ' · on-screen text' : ''}
                {a.grounded_entities?.length ? ` · visible: ${a.grounded_entities.join(', ')}` : ''}
              </span>
            </div>
          ))}
        </div>
      )}
      {data.invalid?.length > 0 && (
        <div style={{ marginTop: 8, fontSize: '0.76rem', color: '#e06c75' }}>
          Referenced but never offered (hallucinated or off-topic): Keyframe {data.invalid.join(', ')}
        </div>
      )}
    </div>
  );
};

/* ─── Custom tooltip for recharts ────────────────────────────────────────────── */
const DarkTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'rgba(10,15,26,0.95)', border: '1px solid rgba(201,168,76,0.25)', borderRadius: 8, padding: '8px 12px', fontSize: '0.74rem', color: '#fdf8ee' }}>
      {label !== undefined && <div style={{ color: 'rgba(201,168,76,0.7)', marginBottom: 4 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || '#c9a84c' }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</div>
      ))}
    </div>
  );
};

/* ─── Main page ──────────────────────────────────────────────────────────────── */
const PHASE_COLORS = ['#c9a84c', '#5ba4cf', '#e06c75', '#98c379'];
const PHASE_LABELS = ['Phase 1: Ingestion', 'Phase 2: Indexing', 'Phase 3: Retrieval', 'Phase 4: Subtitles'];

const PipelineLogs = () => {
  const { taskId } = useParams();
  const [searchParams] = useSearchParams();
  const backTab = searchParams.get('tab');
  const [logs, setLogs] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!taskId) return;
    apiService.getPipelineLogs(taskId)
      .then(data => { setLogs(data.logs); setStatus(data.status); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [taskId]);

  if (loading) return (
    <div className="pl-wrap"><Styles /><div className="pl-loading">Loading pipeline logs…</div></div>
  );
  if (error) return (
    <div className="pl-wrap"><Styles />
      <div className="pl-loading" style={{ color: '#fca5a5' }}>Error: {error}</div>
    </div>
  );

  const p1 = logs?.phase1 || {};
  const p2 = logs?.phase2 || {};
  const p3 = logs?.phase3 || {};

  // Performance bar timings
  const timings = [
    p1.timing_sec ?? 0,
    p2.timing_sec ?? 0,
    p3.timing_sec ?? 0,
    0,  // subtitles not separately tracked yet
  ];
  const totalTime = timings.reduce((a, b) => a + b, 0) || 1;

  // KG nodes / edges for visualiser
  const kgEdges = p2.kg_sample_edges || [];
  const kgNodeIds = new Set();
  kgEdges.forEach(e => { kgNodeIds.add(e.source); kgNodeIds.add(e.target); });
  const kgNodes = Array.from(kgNodeIds).map(id => ({ id }));

  // Cluster scatter data
  const scatterData = (p1.frame_scores_sample || []).map(f => ({
    timestamp: f.timestamp,
    score: f.score,
    cluster: f.cluster,
  }));

  // Chunk size histogram
  const chunkSizes = (p2.chunk_sizes || []).map((size, i) => ({ index: i, tokens: size }));

  // Retrieval scores bar chart
  const retrievalScores = (p3.general_retrieval_scores || []).slice(0, 30).map((s, i) => ({
    name: `C${i + 1}`,
    score: parseFloat((s.score || 0).toFixed(3)),
    vector: parseFloat((s.vector_score || 0).toFixed(3)),
    graph: parseFloat((s.graph_score || 0).toFixed(3)),
  }));

  // Retrieved chunks list
  const retrievedChunks = p3.general_retrieved_chunks || [];

  const fmtTime = (sec) => sec < 60
    ? `${sec.toFixed(1)}s`
    : `${Math.floor(sec / 60)}m ${(sec % 60).toFixed(0)}s`;

  return (
    <div className="pl-wrap">
      <Styles />

      <nav className="pl-nav">
        <a href="/landing" className="pl-nav-brand">Mimir</a>
        <Link
          to={taskId ? `/upload?task=${taskId}${backTab ? `&tab=${backTab}` : ''}` : '/upload'}
          className="pl-back"
        ><ArrowLeft size={14} /> Back to Results</Link>
      </nav>

      <div className="pl-page">
        <div className="pl-title">Pipeline Logs</div>
        <div className="pl-subtitle">
          Task {taskId?.slice(0, 8)}… · Status: <span style={{ color: status === 'completed' ? '#86efac' : '#fca5a5' }}>{status}</span>
          {totalTime > 1 && ` · Total time: ${fmtTime(totalTime)}`}
        </div>

        {/* Performance bar */}
        {totalTime > 1 && (
          <>
            <div className="pl-perf-bar">
              {timings.map((t, i) =>
                t > 0 && (
                  <div key={i} className="pl-perf-seg"
                    style={{ flex: t / totalTime, background: PHASE_COLORS[i], minWidth: t / totalTime > 0.03 ? 0 : 'unset' }}>
                    {t / totalTime > 0.06 ? `${fmtTime(t)}` : ''}
                  </div>
                )
              )}
            </div>
            <div className="pl-perf-legend">
              {PHASE_LABELS.map((label, i) => timings[i] > 0 && (
                <div key={i} className="pl-perf-legend-item">
                  <div className="pl-perf-swatch" style={{ background: PHASE_COLORS[i] }} />
                  {label} ({fmtTime(timings[i])})
                </div>
              ))}
            </div>
          </>
        )}

        {/* Phase 1 */}
        <Panel icon={Zap} title="Phase 1: Ingestion & Preprocessing"
          desc="WhisperX transcription + 1-FPS frame extraction + k-means++ clustering + scene segmentation"
          badge={p1.timing_sec ? fmtTime(p1.timing_sec) : null}>
          <div className="pl-stats">
            <Stat val={p1.transcript_segments} label="Transcript segments" />
            <Stat val={p1.total_frames_at_1fps} label="Frames @ 1FPS" />
            <Stat val={p1.cluster_k} label="k-means k" />
            <Stat val={p1.scene_count} label="Visual scenes" />
            <Stat val={p1.keyframes_selected} label="Keyframes selected" />
          </div>

          {scatterData.length > 0 && (
            <>
              <div className="pl-chart-label">Frame Informativeness by Cluster (scatter)</div>
              <div className="pl-chart-wrap">
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                    <XAxis dataKey="timestamp" name="Time (s)" stroke="rgba(201,168,76,0.3)" tick={{ fill: 'rgba(220,195,130,0.5)', fontSize: 11 }} label={{ value: 'Time (s)', position: 'insideBottom', offset: -12, fill: 'rgba(220,195,130,0.4)', fontSize: 11 }} />
                    <YAxis dataKey="score" name="Score" stroke="rgba(201,168,76,0.3)" tick={{ fill: 'rgba(220,195,130,0.5)', fontSize: 11 }} domain={[0, 1]} />
                    <Tooltip content={<DarkTooltip />} cursor={{ stroke: 'rgba(201,168,76,0.2)' }} />
                    <Scatter data={scatterData} name="Frames">
                      {scatterData.map((d, i) => (
                        <Cell key={i} fill={clusterColor(d.cluster)} fillOpacity={0.75} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                  {[...new Set(scatterData.map(d => d.cluster))].slice(0, 12).map(c => (
                    <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.66rem', color: 'rgba(220,195,130,0.55)' }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: clusterColor(c) }} />
                      Cluster {c}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </Panel>

        {/* Phase 2 */}
        <Panel icon={GitBranch} title="Phase 2: Dual-Channel Indexing"
          desc="Greedy semantic chunking + spaCy Knowledge Graph + FAISS vector store"
          badge={p2.timing_sec ? fmtTime(p2.timing_sec) : null}>
          <div className="pl-stats">
            <Stat val={p2.total_chunks} label="Semantic chunks" />
            <Stat val={p2.kg_nodes} label="KG nodes" />
            <Stat val={p2.kg_edges} label="KG edges" />
            <Stat val={p2.vector_index_size} label="Vector index size" />
          </div>

          {chunkSizes.length > 0 && (
            <>
              <div className="pl-chart-label">Chunk Size Distribution (tokens)</div>
              <div className="pl-chart-wrap">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={chunkSizes} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                    <XAxis dataKey="index" hide />
                    <YAxis stroke="rgba(201,168,76,0.3)" tick={{ fill: 'rgba(220,195,130,0.5)', fontSize: 10 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Bar dataKey="tokens" fill="rgba(91,164,207,0.65)" radius={[2, 2, 0, 0]} name="Tokens" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {kgNodes.length > 0 && (
            <>
              <div className="pl-chart-label">Knowledge Graph (top {Math.min(kgNodes.length, 80)} nodes — hover to label)</div>
              <div className="pl-chart-wrap" style={{ padding: '16px' }}>
                <KGGraph nodes={kgNodes} edges={kgEdges} />
              </div>
            </>
          )}
        </Panel>

        {/* Phase 3 */}
        <Panel icon={BarChart2} title="Phase 3: Hybrid Retrieval + Summarization"
          desc="Fork-join KG + vector search → fusion → Gemini (retrieved chunks only)"
          badge={p3.timing_sec ? fmtTime(p3.timing_sec) : null}>
          <div className="pl-stats">
            <Stat val={p3.general_retrieved_count} label="Chunks retrieved" />
            <Stat val={p3.general_top_k} label="Budget (k)" />
            <Stat val={p3.topic_retrieved_count || null} label="Topic retrieved" />
            <Stat val={retrievalScores.length} label="Scores tracked" />
          </div>

          {p3.general_query && (
            <div style={{ marginBottom: 18, padding: '10px 14px', background: 'rgba(5,10,20,0.45)', border: '1px solid rgba(201,168,76,0.08)', borderRadius: 10 }}>
              <div style={{ fontSize: '0.66rem', color: 'rgba(201,168,76,0.55)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>Query</div>
              <div style={{ fontSize: '0.8rem', color: 'rgba(248,235,190,0.75)' }}>{p3.general_query}</div>
            </div>
          )}

          {retrievalScores.length > 0 && (
            <>
              <div className="pl-chart-label">Retrieval Relevance Scores (top 30)</div>
              <div className="pl-chart-wrap">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={retrievalScores} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                    <XAxis dataKey="name" stroke="rgba(201,168,76,0.3)" tick={{ fill: 'rgba(220,195,130,0.5)', fontSize: 9 }} />
                    <YAxis domain={[0, 1]} stroke="rgba(201,168,76,0.3)" tick={{ fill: 'rgba(220,195,130,0.5)', fontSize: 10 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Bar dataKey="vector" stackId="a" fill="rgba(91,164,207,0.75)" name="Vector" />
                    <Bar dataKey="graph" stackId="a" fill="rgba(201,168,76,0.65)" name="Graph" />
                    <Legend wrapperStyle={{ fontSize: '0.72rem', color: 'rgba(220,195,130,0.55)' }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {retrievedChunks.length > 0 && (
            <>
              <div className="pl-chart-label">Retrieved Chunks (sorted by time)</div>
              <div className="pl-chunk-list">
                {retrievedChunks.map((c, i) => {
                  const ts = c.time_start ?? 0;
                  const m = Math.floor(ts / 60), s = Math.floor(ts % 60);
                  const label = `${m}:${String(s).padStart(2, '0')}`;
                  return (
                    <div key={i} className="pl-chunk-item">
                      <span className="pl-chunk-ts">{label}</span>
                      <span className="pl-chunk-score">{(c.score ?? 0).toFixed(3)}</span>
                      <span className="pl-chunk-text">{c.text || '—'}</span>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </Panel>

        {/* Keyframe references */}
        <Panel icon={BarChart2} title="Keyframe References"
          desc="Which keyframes were fed to the summaries, and which the LLM actually cited"
          badge={p3.keyframe_refs ? `${p3.keyframe_refs.general?.cited_count ?? 0}/${p3.keyframe_refs.general?.offered_count ?? 0} cited` : null}>
          {!p3.keyframe_refs ? (
            <div style={{ fontSize: '0.8rem', color: 'rgba(220,195,130,0.5)', fontStyle: 'italic' }}>
              No keyframe-reference data for this task (processed before this feature existed).
            </div>
          ) : (
            <>
              <div className="pl-stats">
                <Stat val={p3.keyframe_refs.total_keyframes} label="Keyframes" />
                <Stat val={p3.keyframe_refs.hosting_chunks} label="Chunks hosting a keyframe" />
                <Stat val={p3.keyframe_labels ?? 0} label="LLM labels" />
                <Stat val={p2.visual?.frames_with_ocr ?? '—'} label="Frames with OCR" />
                <Stat val={p2.visual?.graph?.grounded_links ?? '—'} label="CLIP grounded links" />
                <Stat val={p2.visual?.graph?.merged_with_speech ?? '—'} label="Said + shown entities" />
              </div>
              <RefBlock title="General summary" data={p3.keyframe_refs.general} />
              <RefBlock title={`Topic summary${p3.topic ? `: ${p3.topic}` : ''}`} data={p3.keyframe_refs.topic} />
            </>
          )}
        </Panel>

        {/* Explanation */}
        <Panel icon={Clock} title="Architecture Explanation" desc="Why each stage exists" defaultOpen={false}>
          {[
            ['Phase 1: Ingestion', 'WhisperX extracts word-level timestamps. OpenCV samples 1 frame/second. A three-metric score (sharpness, contrast, edge density) rates each frame. K-means++ clusters frames by pixel features with k proportional to video duration — this discovers visual scene changes without supervision. Per-scene keyframes are selected and deduplicated by perceptual hash.'],
            ['Phase 2: Dual-Channel Indexing', 'Transcript sentences are embedded with all-MiniLM-L6-v2. A greedy algorithm starts a new chunk when cosine similarity between consecutive sentences drops below 0.75 or the token count exceeds 400. Each chunk feeds both a NetworkX knowledge graph (spaCy NER + SVO relations with timestamp metadata) and a FAISS IndexFlatIP (cosine similarity via normalised inner product).'],
            ['Phase 3: Hybrid Retrieval', 'A query is decomposed into entity terms (graph channel) and embedded (vector channel) simultaneously. Graph search returns chunks containing matched entities weighted by mention frequency. Vector search returns top-k cosine-similar chunks. Results are fused with final_score = 0.5 × vector + 0.3 × graph/10 + 0.2 × text_overlap, then ranked and the top-k (k scales with video length: 15–60 general, 8–30 topic) fed to Gemini, trimmed to a context budget. The LLM never sees the full transcript — only the retrieved context.'],
            ['Two Summary Types', 'General summary uses the query "summarize the main topics, key insights, and important content" through the same hybrid retrieval. Topic summary uses the user-provided topic string as the query. Both go through the identical retrieval + Gemini pipeline; the only difference is the query.'],
          ].map(([title, body]) => (
            <div key={title} style={{ marginBottom: 20 }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 500, color: '#c9a84c', marginBottom: 6 }}>{title}</div>
              <div style={{ fontSize: '0.8rem', color: 'rgba(248,235,190,0.72)', lineHeight: 1.75 }}>{body}</div>
            </div>
          ))}
        </Panel>
      </div>
    </div>
  );
};

export default PipelineLogs;
