import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Download, Play, FileText, Brain, Image, Search, CheckCircle, BarChart2,
} from 'lucide-react';

/* ─── Self-contained styles ──────────────────────────────────────────────────── */
const PanelStyles = () => (
  <style>{`
    .rp-wrap {
      animation: rpRiseIn 0.55s cubic-bezier(0.23,1,0.32,1) both;
      font-family: 'DM Sans', sans-serif;
    }
    @keyframes rpRiseIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Success bar */
    .rp-success-bar {
      display: flex; align-items: center; gap: 14px;
      background: rgba(34,197,94,0.05); border: 1px solid rgba(34,197,94,0.18);
      border-radius: 14px; padding: 14px 20px; margin-bottom: 24px;
    }
    .rp-success-icon {
      width: 38px; height: 38px; border-radius: 50%;
      background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.2);
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .rp-success-title { font-size: 0.92rem; font-weight: 500; color: #86efac; }
    .rp-success-sub { font-size: 0.78rem; color: rgba(134,239,172,0.65); margin-top: 2px; }

    /* Tab bar */
    .rp-tabs {
      display: flex; gap: 4px;
      background: rgba(5, 10, 20, 0.55);
      border: 1px solid rgba(201, 168, 76, 0.18);
      border-bottom: none;
      border-radius: 18px 18px 0 0;
      padding: 6px 6px 0 6px;
      overflow-x: auto;
    }
    .rp-tabs::-webkit-scrollbar { height: 0; }

    .rp-tab-btn {
      font-family: 'DM Sans', sans-serif; font-size: 0.8rem; font-weight: 500;
      padding: 9px 18px; border-radius: 12px 12px 0 0;
      border: none; background: transparent; color: rgba(220, 195, 130, 0.5);
      cursor: pointer; white-space: nowrap;
      display: flex; align-items: center; gap: 7px;
      transition: all 0.22s ease; position: relative;
      border-bottom: 2px solid transparent;
    }
    .rp-tab-btn:hover { color: rgba(248, 235, 190, 0.82); background: rgba(201, 168, 76, 0.05); }
    .rp-tab-btn.active {
      color: #fdf8ee;
      background: rgba(20, 35, 55, 0.50);
      border-bottom-color: #c9a84c;
    }
    .rp-tab-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: #c9a84c; opacity: 0;
      transition: opacity 0.2s;
    }
    .rp-tab-btn.active .rp-tab-dot { opacity: 1; }

    /* Panel */
    .rp-panel {
      background: rgba(20, 35, 55, 0.50);
      backdrop-filter: blur(28px) saturate(160%);
      -webkit-backdrop-filter: blur(28px) saturate(160%);
      border: 1px solid rgba(201, 168, 76, 0.18);
      border-top: none;
      border-radius: 0 0 28px 28px;
      overflow: hidden;
      min-height: 360px;
    }
    .rp-panel-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 20px 26px;
      border-bottom: 1px solid rgba(201, 168, 76, 0.08);
    }
    .rp-panel-title {
      display: flex; align-items: center; gap: 10px;
      font-family: 'DM Serif Display', Georgia, serif; font-size: 1.1rem; color: #fdf8ee;
    }
    .rp-panel-title-icon {
      width: 34px; height: 34px; border-radius: 10px;
      background: rgba(201, 168, 76, 0.1);
      border: 1px solid rgba(201, 168, 76, 0.22);
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .rp-panel-actions { display: flex; align-items: center; gap: 8px; }

    .rp-icon-btn {
      width: 34px; height: 34px; border-radius: 9px;
      background: rgba(201, 168, 76, 0.07);
      border: 1px solid rgba(201, 168, 76, 0.18);
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; color: #c9a84c; transition: all 0.2s;
    }
    .rp-icon-btn:hover { background: rgba(201, 168, 76, 0.16); border-color: rgba(201, 168, 76, 0.45); }

    .rp-panel-body { padding: 24px 26px; }

    /* Transcript */
    .rp-transcript-body {
      background: rgba(5, 10, 20, 0.55);
      border: 1px solid rgba(201, 168, 76, 0.08);
      border-radius: 14px;
      padding: 20px 22px;
      max-height: 420px; overflow-y: auto;
      font-family: 'Menlo', 'Consolas', monospace;
      font-size: 0.82rem; color: rgba(248, 235, 190, 0.82);
      line-height: 1.8; letter-spacing: 0.01em;
      white-space: pre-wrap; word-wrap: break-word;
    }

    /* Video */
    .rp-video-container {
      border-radius: 14px; overflow: hidden;
      background: #000;
      border: 1px solid rgba(201, 168, 76, 0.18);
      aspect-ratio: 16 / 9; position: relative;
    }
    .rp-video-container video { width: 100%; height: 100%; display: block; outline: none; }
    .rp-video-fallback {
      display: flex; align-items: center; justify-content: center; height: 100%;
      color: rgba(220, 195, 130, 0.5); font-size: 0.84rem;
    }

    .rp-srt-toggle-row { margin-top: 14px; display: flex; align-items: center; gap: 8px; }
    .rp-srt-toggle-btn {
      font-family: 'DM Sans', sans-serif; font-size: 0.76rem; color: rgba(201, 168, 76, 0.55);
      cursor: pointer; background: none; border: none; padding: 0;
      transition: color 0.2s; display: inline-flex; align-items: center; gap: 6px;
    }
    .rp-srt-toggle-btn:hover { color: #c9a84c; }
    .rp-srt-raw-box {
      margin-top: 12px; max-height: 180px; overflow-y: auto;
      background: rgba(5, 10, 20, 0.6);
      border: 1px solid rgba(201, 168, 76, 0.18);
      border-radius: 10px; padding: 14px;
    }
    .rp-srt-raw-box pre {
      font-family: 'Menlo', 'Consolas', monospace; font-size: 0.76rem;
      color: rgba(248, 235, 190, 0.82); white-space: pre-wrap; line-height: 1.6;
    }

    /* Caption style selector */
    .rp-cap-style-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
    .rp-cap-style-label { font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.16em; color: rgba(201,168,76,0.55); }
    .rp-cap-seg { display: inline-flex; padding: 3px; background: rgba(5,10,20,0.6); border: 1px solid rgba(201,168,76,0.18); border-radius: 100px; gap: 2px; }
    .rp-cap-seg-btn {
      font-family: 'DM Sans', sans-serif; font-size: 0.73rem; font-weight: 500;
      padding: 5px 13px; border-radius: 100px; border: none; background: transparent;
      color: rgba(220,195,130,0.5); cursor: pointer; transition: all 0.2s ease;
    }
    .rp-cap-seg-btn.active {
      background: linear-gradient(135deg, rgba(201,168,76,0.9) 0%, rgba(170,130,40,0.9) 100%);
      color: #0a0f1a; box-shadow: 0 2px 10px rgba(201,168,76,0.2);
    }

    /* Caption overlay (in-app karaoke / flashy styles) */
    .rp-cap-overlay { position: absolute; inset: 0; pointer-events: none; display: flex; align-items: flex-end; justify-content: center; padding: 0 6% 7%; z-index: 2; }
    .rp-cap { max-width: 94%; text-align: center; line-height: 1.32; }
    .rp-cap span { display: inline-block; }

    .rp-cap--classic {
      font-family: 'DM Sans', sans-serif; font-size: clamp(0.9rem, 2.6vw, 1.4rem); font-weight: 600;
      color: #fff; text-shadow: 0 2px 6px rgba(0,0,0,0.9); background: rgba(0,0,0,0.5); padding: 6px 14px; border-radius: 8px;
    }

    .rp-cap--karaoke {
      font-family: 'DM Sans', sans-serif; font-size: clamp(0.95rem, 2.8vw, 1.5rem); font-weight: 700;
      text-shadow: 0 2px 8px rgba(0,0,0,0.95); background: rgba(0,0,0,0.42); padding: 8px 16px; border-radius: 10px;
    }
    .rp-cap--karaoke span { color: rgba(255,255,255,0.5); padding: 0 4px; transition: color 0.12s ease; }
    .rp-cap--karaoke span.spoken { color: #fff; }
    .rp-cap--karaoke span.active { color: #f0d070; text-shadow: 0 0 14px rgba(240,208,112,0.85); }

    .rp-cap--clean {
      font-family: 'DM Sans', sans-serif; font-size: clamp(0.85rem, 2.3vw, 1.25rem); font-weight: 500;
      color: rgba(255,255,255,0.9); text-shadow: 0 1px 5px rgba(0,0,0,0.9);
      background: linear-gradient(transparent, rgba(0,0,0,0.55)); padding: 10px 18px; border-radius: 6px;
    }
    .rp-cap--clean span { padding: 0 3px; transition: color 0.12s ease; }
    .rp-cap--clean span.active { color: #fff; font-weight: 700; }

    /* Summary */
    .rp-summary-layout { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 640px) { .rp-summary-layout { grid-template-columns: 3fr 2fr; } }

    .rp-summary-text-block {
      background: rgba(5, 10, 20, 0.55);
      border: 1px solid rgba(201, 168, 76, 0.08);
      border-radius: 14px; padding: 20px 22px;
      font-size: 0.88rem; color: rgba(248, 235, 190, 0.82); line-height: 1.8; font-weight: 300;
      max-height: 360px; overflow-y: auto;
      white-space: pre-wrap; word-wrap: break-word;
    }

    .rp-keypoints-block {
      background: rgba(5, 10, 20, 0.55);
      border: 1px solid rgba(201, 168, 76, 0.08);
      border-radius: 14px; padding: 18px 20px;
      max-height: 360px; overflow-y: auto;
    }
    .rp-keypoints-label {
      font-size: 0.6rem; font-weight: 500; letter-spacing: 0.2em;
      text-transform: uppercase; color: rgba(201, 168, 76, 0.55);
      margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
    }
    .rp-keypoints-label::before { content: ''; width: 4px; height: 4px; border-radius: 50%; background: rgba(201, 168, 76, 0.55); }

    .rp-keypoint-item {
      display: flex; align-items: flex-start; gap: 10px;
      padding: 9px 0; border-bottom: 1px solid rgba(201, 168, 76, 0.07);
      font-size: 0.82rem; color: rgba(248, 235, 190, 0.82); line-height: 1.55;
    }
    .rp-keypoint-item:last-child { border-bottom: none; padding-bottom: 0; }
    .rp-keypoint-bullet {
      width: 18px; height: 18px; border-radius: 50%; flex-shrink: 0; margin-top: 1px;
      background: rgba(201, 168, 76, 0.12);
      border: 1px solid rgba(201, 168, 76, 0.28);
      display: flex; align-items: center; justify-content: center;
      font-size: 0.6rem; font-weight: 700; color: #c9a84c; font-family: 'Menlo', monospace;
    }

    /* Badges */
    .rp-meta-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .rp-badge {
      font-size: 0.6rem; font-weight: 500; letter-spacing: 0.12em;
      text-transform: uppercase; padding: 3px 10px; border-radius: 100px;
    }
    .rp-badge-teal {
      background: rgba(201, 168, 76, 0.1);
      color: #c9a84c;
      border: 1px solid rgba(201, 168, 76, 0.22);
    }
    .rp-badge-dim {
      background: rgba(220, 195, 130, 0.06);
      color: rgba(220, 195, 130, 0.5);
      border: 1px solid rgba(201, 168, 76, 0.18);
    }

    /* Keyframes */
    .rp-kf-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); gap: 10px;
    }
    .rp-kf-cell {
      position: relative; border-radius: 10px; overflow: hidden;
      border: 1px solid rgba(201, 168, 76, 0.18);
      background: rgba(0, 0, 0, 0.5);
      aspect-ratio: 16/9; cursor: pointer; text-decoration: none;
      transition: all 0.25s cubic-bezier(0.23, 1, 0.32, 1);
    }
    .rp-kf-cell:hover {
      transform: scale(1.04);
      border-color: rgba(201, 168, 76, 0.5);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
    }
    .rp-kf-cell img {
      width: 100%; height: 100%; object-fit: cover; display: block;
      transition: transform 0.4s ease;
    }
    .rp-kf-cell:hover img { transform: scale(1.06); }
    .rp-kf-time {
      position: absolute; bottom: 6px; left: 6px;
      background: rgba(10, 15, 26, 0.88); backdrop-filter: blur(8px);
      border: 1px solid rgba(201, 168, 76, 0.18);
      color: #c9a84c;
      font-family: 'Menlo','Consolas',monospace; font-size: 0.68rem; font-weight: 500;
      padding: 2px 7px; border-radius: 100px;
    }
    .rp-kf-empty {
      padding: 48px 16px; text-align: center;
      color: rgba(220, 195, 130, 0.5); font-style: italic; font-size: 0.86rem; font-weight: 300;
    }

    /* RAG */
    .rp-rag-body {
      background: rgba(5, 10, 20, 0.55);
      border: 1px solid rgba(201, 168, 76, 0.08);
      border-radius: 14px; padding: 20px 22px;
      font-size: 0.88rem; color: rgba(248, 235, 190, 0.82); line-height: 1.8; font-weight: 300;
      max-height: 420px; overflow-y: auto;
      white-space: pre-wrap; word-wrap: break-word;
    }
  `}</style>
);

/* ─── Caption styles + overlay (in-app karaoke / flashy captions) ─────────────── */
const CAPTION_STYLES = [
  { id: 'classic', label: 'Classic' },
  { id: 'karaoke', label: 'Karaoke' },
  { id: 'clean',   label: 'Clean'   },
];

// Split a segment into per-word timings. Uses real word timestamps if present
// (seg.words), otherwise interpolates within [start,end] by word length — zero
// backend cost, accurate enough to read along for short Whisper segments.
function wordsForSegment(seg) {
  if (!seg) return [];
  if (Array.isArray(seg.words) && seg.words.length) {
    return seg.words.map(w => ({ text: (w.word ?? w.text ?? '').trim(), start: w.start, end: w.end }));
  }
  const toks = (seg.text || '').trim().split(/\s+/).filter(Boolean);
  const total = toks.reduce((a, w) => a + w.length + 1, 0) || 1;
  const dur = Math.max((seg.end ?? 0) - (seg.start ?? 0), 0.01);
  let acc = seg.start ?? 0;
  return toks.map(w => {
    const wd = (dur * (w.length + 1)) / total;
    const o = { text: w, start: acc, end: acc + wd };
    acc += wd;
    return o;
  });
}

const CaptionOverlay = ({ videoRef, segments, styleId }) => {
  const [t, setT] = useState(0);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let raf;
    const tick = () => { setT(v.currentTime); raf = requestAnimationFrame(tick); };
    const start = () => { cancelAnimationFrame(raf); tick(); };
    const stop = () => cancelAnimationFrame(raf);
    const seek = () => setT(v.currentTime);
    v.addEventListener('play', start);
    v.addEventListener('pause', stop);
    v.addEventListener('ended', stop);
    v.addEventListener('seeking', seek);
    if (!v.paused && !v.ended) start(); else setT(v.currentTime);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener('play', start);
      v.removeEventListener('pause', stop);
      v.removeEventListener('ended', stop);
      v.removeEventListener('seeking', seek);
    };
  }, [videoRef, segments]);

  const seg = (segments || []).find(s => t >= (s.start ?? 0) && t < (s.end ?? 0));
  const words = useMemo(() => wordsForSegment(seg), [seg]);
  if (!seg || !words.length) return null;

  const activeIdx = words.findIndex(w => t >= w.start && t < w.end);

  return (
    <div className={`rp-cap-overlay rp-cap-overlay--${styleId}`}>
      <div className={`rp-cap rp-cap--${styleId}`}>
        {styleId === 'classic'
          ? seg.text
          : words.map((w, i) => {
              const cls = i === activeIdx ? 'active' : (activeIdx >= 0 && i < activeIdx ? 'spoken' : '');
              return <span key={i} className={cls}>{w.text}{' '}</span>;
            })}
      </div>
    </div>
  );
};

/* ─── ResultsPanel component ─────────────────────────────────────────────────── */
const ResultsPanel = ({
  results,
  selectedOptions,
  subtitleFormat = 'srt',
  videoFile,
  onDownload,
  showRawSrt,
  setShowRawSrt,
  showSuccessBar = true,
  taskId,
}) => {
  const tabs = [];
  if (selectedOptions.transcript) tabs.push({ id: 'transcript', label: 'Transcript',      icon: FileText });
  if (selectedOptions.subtitles)  tabs.push({ id: 'subtitles',  label: 'Subtitles',       icon: Play });
  if (selectedOptions.summary)    tabs.push({ id: 'summary',    label: 'General Summary', icon: Brain });
  // Topic summary tab — always show if enabled, dim when no data
  if (selectedOptions.topicBased) tabs.push({ id: 'topic',      label: 'Topic Summary',   icon: Search });
  // Keyframes always extracted but tab shown when data is present
  if (results.keyframes?.length > 0) tabs.push({ id: 'keyframes', label: 'Keyframes', icon: Image });

  const [activeTab, setActiveTab] = useState(tabs[0]?.id || 'transcript');
  const [captionStyle, setCaptionStyle] = useState('classic');
  const videoRef = useRef(null);
  const captionSegments = results.segments || [];

  return (
    <div className="rp-wrap">
      <PanelStyles />

      {showSuccessBar && (
        <div className="rp-success-bar">
          <div className="rp-success-icon">
            <CheckCircle size={18} color="#22c55e" />
          </div>
          <div style={{ flex: 1 }}>
            <div className="rp-success-title">Processing complete</div>
            <div className="rp-success-sub">{videoFile?.name || 'Result'} — all outputs are ready</div>
          </div>
          {taskId && (
            <Link
              to={`/logs/${taskId}`}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                fontSize: '0.76rem', color: '#c9a84c', textDecoration: 'none',
                padding: '6px 14px', borderRadius: 8,
                background: 'rgba(201,168,76,0.07)',
                border: '1px solid rgba(201,168,76,0.22)',
                flexShrink: 0, whiteSpace: 'nowrap',
              }}
            >
              <BarChart2 size={13} /> Pipeline Logs
            </Link>
          )}
        </div>
      )}

      {/* Tab bar */}
      <div className="rp-tabs">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`rp-tab-btn${activeTab === id ? ' active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            <div className="rp-tab-dot" />
            <Icon size={13} />
            {label}
          </button>
        ))}
      </div>

      {/* Panel */}
      <div className="rp-panel">

        {/* TRANSCRIPT */}
        {activeTab === 'transcript' && (
          <>
            <div className="rp-panel-header">
              <div className="rp-panel-title">
                <div className="rp-panel-title-icon"><FileText size={15} color="rgba(201,168,76,0.85)" /></div>
                Full Transcript
              </div>
              <div className="rp-panel-actions">
                <button onClick={() => onDownload('transcript')} className="rp-icon-btn" title="Download transcript">
                  <Download size={14} />
                </button>
              </div>
            </div>
            <div className="rp-panel-body">
              <div className="rp-transcript-body">{results.transcript}</div>
            </div>
          </>
        )}

        {/* SUBTITLES */}
        {activeTab === 'subtitles' && (
          <>
            <div className="rp-panel-header">
              <div className="rp-panel-title">
                <div className="rp-panel-title-icon"><Play size={15} color="rgba(201,168,76,0.85)" /></div>
                Video with Subtitles
              </div>
              <div className="rp-panel-actions">
                <span className="rp-badge rp-badge-dim">{(subtitleFormat || 'srt').toUpperCase()}</span>
                <button onClick={() => onDownload('subtitles')} className="rp-icon-btn" title="Download subtitles">
                  <Download size={14} />
                </button>
                {results.subtitles_vtt_url && taskId && (
                  <a href={`/api/v1/download/subtitled-video/${taskId}`} download
                    className="rp-icon-btn" title="Download video with subtitles">
                    <Play size={14} />
                  </a>
                )}
              </div>
            </div>
            <div className="rp-panel-body">
              {results.video_url && captionSegments.length > 0 && (
                <div className="rp-cap-style-row">
                  <span className="rp-cap-style-label">Caption style</span>
                  <div className="rp-cap-seg">
                    {CAPTION_STYLES.map(({ id, label }) => (
                      <button key={id} type="button"
                        onClick={() => setCaptionStyle(id)}
                        className={`rp-cap-seg-btn${captionStyle === id ? ' active' : ''}`}>
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="rp-video-container">
                {results.video_url ? (
                  <>
                  <video ref={videoRef} controls preload="metadata">
                    <source src={results.video_url} />
                    {/* Native track only as fallback when we have no timed segments to
                        drive the custom overlay — avoids double captions. */}
                    {results.subtitles_vtt_url && captionSegments.length === 0 && (
                      <track kind="subtitles" label="English" srcLang="en"
                        src={results.subtitles_vtt_url} default />
                    )}
                    Your browser does not support inline video playback.
                  </video>
                  {captionSegments.length > 0 && (
                    <CaptionOverlay videoRef={videoRef} segments={captionSegments} styleId={captionStyle} />
                  )}
                  </>
                ) : (
                  <div className="rp-video-fallback">Video preview not available.</div>
                )}
              </div>
              <div className="rp-srt-toggle-row">
                <button type="button" onClick={() => setShowRawSrt(v => !v)} className="rp-srt-toggle-btn">
                  {showRawSrt ? '▾' : '▸'} {showRawSrt ? 'Hide' : 'Show'} raw subtitles
                </button>
              </div>
              {showRawSrt && (
                <div className="rp-srt-raw-box"><pre>{results.subtitles}</pre></div>
              )}
            </div>
          </>
        )}

        {/* SUMMARY */}
        {activeTab === 'summary' && results.summary && (
          <>
            <div className="rp-panel-header">
              <div className="rp-panel-title">
                <div className="rp-panel-title-icon"><Brain size={15} color="rgba(201,168,76,0.85)" /></div>
                General Summary
              </div>
              <div className="rp-panel-actions">
                <div className="rp-meta-row">
                  {results.summary.retrieved_chunk_count != null && (
                    <span className="rp-badge rp-badge-dim">{results.summary.retrieved_chunk_count} chunks retrieved</span>
                  )}
                  <span className={`rp-badge ${results.summary.summary_type === 'gemini' ? 'rp-badge-teal' : 'rp-badge-dim'}`}>
                    {results.summary.summary_type === 'gemini' ? 'Gemini' : 'Fallback'}
                  </span>
                  <button onClick={() => onDownload('summary')} className="rp-icon-btn" title="Download summary">
                    <Download size={14} />
                  </button>
                </div>
              </div>
            </div>
            <div className="rp-panel-body">
              <div className="rp-summary-layout">
                <div className="rp-summary-text-block">
                  {results.summary.text || results.summary.summary || ''}
                </div>
                {results.summary.key_points?.length > 0 && (
                  <div className="rp-keypoints-block">
                    <div className="rp-keypoints-label">Key Points</div>
                    {results.summary.key_points.map((point, i) => (
                      <div key={i} className="rp-keypoint-item">
                        <div className="rp-keypoint-bullet">{i + 1}</div>
                        <span>{point}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* KEYFRAMES */}
        {activeTab === 'keyframes' && (
          <>
            <div className="rp-panel-header">
              <div className="rp-panel-title">
                <div className="rp-panel-title-icon"><Image size={15} color="rgba(201,168,76,0.85)" /></div>
                Keyframes
                {results.keyframes?.length > 0 && (
                  <span className="rp-badge rp-badge-dim" style={{ marginLeft: 8 }}>{results.keyframes.length} frames</span>
                )}
              </div>
              <div className="rp-panel-actions">
                <button onClick={() => onDownload('keyframes')} className="rp-icon-btn" title="Download timestamp list">
                  <Download size={14} />
                </button>
              </div>
            </div>
            <div className="rp-panel-body">
              {!results.keyframes || results.keyframes.length === 0 ? (
                <div className="rp-kf-empty">No keyframes extracted.</div>
              ) : (
                <div className="rp-kf-grid">
                  {results.keyframes.map((kf, i) => {
                    const url = kf.frame_url || null;
                    const ts = typeof kf.timestamp === 'number'
                      ? `${Math.floor(kf.timestamp / 60)}:${String(Math.floor(kf.timestamp % 60)).padStart(2, '0')}`
                      : (kf.timestamp || `${i + 1}`);
                    return url ? (
                      <a key={i} href={url} target="_blank" rel="noreferrer" className="rp-kf-cell">
                        <img src={url} alt={`Keyframe at ${ts}`} loading="lazy" />
                        <span className="rp-kf-time">{ts}</span>
                      </a>
                    ) : (
                      <div key={i} className="rp-kf-cell" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <span className="rp-kf-time" style={{ position: 'static' }}>{ts}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}

        {/* TOPIC SUMMARY */}
        {activeTab === 'topic' && (
          <>
            <div className="rp-panel-header">
              <div className="rp-panel-title">
                <div className="rp-panel-title-icon"><Search size={15} color="rgba(201,168,76,0.85)" /></div>
                Topic Summary
                {results.topic_summary?.topic && (
                  <span style={{ fontSize: '0.72rem', color: 'rgba(220,195,130,0.5)', fontFamily: 'DM Sans, sans-serif', marginLeft: 8 }}>
                    {results.topic_summary.topic}
                  </span>
                )}
              </div>
              <div className="rp-panel-actions">
                <div className="rp-meta-row">
                  {results.topic_summary?.retrieved_chunk_count != null && (
                    <span className="rp-badge rp-badge-dim">{results.topic_summary.retrieved_chunk_count} chunks retrieved</span>
                  )}
                  <span className={`rp-badge ${results.topic_summary?.summary_type === 'gemini' ? 'rp-badge-teal' : 'rp-badge-dim'}`}>
                    {results.topic_summary?.summary_type === 'gemini' ? 'Gemini' : 'Fallback'}
                  </span>
                  <button onClick={() => onDownload('topic_summary')} className="rp-icon-btn" title="Download topic summary">
                    <Download size={14} />
                  </button>
                </div>
              </div>
            </div>
            <div className="rp-panel-body">
              {results.topic_summary ? (
                <div className="rp-summary-layout">
                  <div className="rp-summary-text-block">
                    {results.topic_summary.text || 'No topic summary available.'}
                  </div>
                  {results.topic_summary.key_points?.length > 0 && (
                    <div className="rp-keypoints-block">
                      <div className="rp-keypoints-label">Key Points</div>
                      {results.topic_summary.key_points.map((point, i) => (
                        <div key={i} className="rp-keypoint-item">
                          <div className="rp-keypoint-bullet">{i + 1}</div>
                          <span>{point}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ padding: '48px 24px', textAlign: 'center', color: 'rgba(220,195,130,0.5)', fontSize: '0.86rem', fontStyle: 'italic' }}>
                  No topic summary — enable "Topic Summary" and enter a topic before processing.
                </div>
              )}
            </div>
          </>
        )}

      </div>
    </div>
  );
};

export default ResultsPanel;
