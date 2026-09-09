import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Upload, FileVideo, Clock, FileText, Download, Settings,
  CheckCircle, AlertCircle, RotateCcw, X, ChevronRight,
  Brain, Languages, Image, Search, Play, Zap
} from 'lucide-react';
import { apiService } from '../services/api';
import ResultsPanel from '../components/ResultsPanel';

/* ─── Styles ─────────────────────────────────────────────────────────────────── */
const PageStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&family=Cinzel+Decorative:wght@700&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }

    :root {
      --bg:           #0a0f1a;
      --bg-mid:       #0d1520;
      --bg-light:     #111d2a;
      --glass-bg:     rgba(20, 35, 55, 0.50);
      --glass-border: rgba(201, 168, 76, 0.18);
      --glass-shine:  rgba(240, 210, 130, 0.07);
      --gold:         #c9a84c;
      --gold-bright:  #f0d070;
      --gold-dim:     rgba(201, 168, 76, 0.55);
      --gold-faint:   rgba(201, 168, 76, 0.14);
      --white:        #fdf8ee;
      --text:         rgba(248, 235, 190, 0.82);
      --text-dim:     rgba(220, 195, 130, 0.5);
      --font-serif:   'DM Serif Display', Georgia, serif;
      --font-sans:    'DM Sans', sans-serif;
      --font-brand:   'Cinzel Decorative', cursive;
      --radius-lg:    20px;
      --radius-xl:    28px;
    }

    .up-wrap {
      font-family: var(--font-sans);
      background: var(--bg);
      min-height: 100vh;
      color: var(--white);
      position: relative;
      overflow-x: hidden;
    }

    .up-mesh {
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image:
        linear-gradient(rgba(201,168,76,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(201,168,76,0.025) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: radial-gradient(ellipse 80% 60% at 50% 30%, black 10%, transparent 80%);
    }

    .up-orb { position: fixed; border-radius: 50%; pointer-events: none; filter: blur(80px); z-index: 0; }
    .up-orb-1 {
      width: 500px; height: 500px;
      background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
      top: -100px; right: 0%;
      animation: orbDrift 14s ease-in-out infinite;
    }
    .up-orb-2 {
      width: 380px; height: 380px;
      background: radial-gradient(circle, rgba(160,120,40,0.07) 0%, transparent 70%);
      bottom: 5%; left: -80px;
      animation: orbDrift 18s ease-in-out infinite reverse;
    }
    @keyframes orbDrift {
      0%,100% { transform: translate(0,0) scale(1); }
      33% { transform: translate(14px,-20px) scale(1.03); }
      66% { transform: translate(-10px,10px) scale(0.97); }
    }

    /* ── Nav ── */
    .up-nav {
      position: sticky; top: 0; z-index: 100;
      height: 66px; display: flex; align-items: center;
      padding: 0 clamp(20px, 5vw, 72px);
      justify-content: space-between;
      background: rgba(10, 15, 26, 0.78);
      backdrop-filter: blur(28px) saturate(180%);
      -webkit-backdrop-filter: blur(28px) saturate(180%);
      border-bottom: 1px solid var(--glass-border);
    }
    .up-nav-brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
    .up-nav-brand img { width: 36px; height: 36px; border-radius: 50%; border: 1.5px solid rgba(201,168,76,0.35); }
    .up-nav-brand-name { font-family: var(--font-brand); font-size: 1.2rem; color: var(--gold); letter-spacing: 0.02em; }
    .up-nav-links { display: flex; align-items: center; gap: 2rem; }
    .up-nav-a { font-size: 0.82rem; font-weight: 500; letter-spacing: 0.04em; color: var(--text); text-decoration: none; transition: color 0.25s; }
    .up-nav-a:hover { color: var(--gold); }

    /* ── Page body ── */
    .up-page {
      position: relative; z-index: 1;
      max-width: 1060px; margin: 0 auto;
      padding: clamp(48px,7vw,96px) clamp(20px,5vw,48px) 80px;
    }

    /* ── Page header ── */
    .up-header {
      text-align: center; margin-bottom: 52px;
      animation: riseIn 0.8s cubic-bezier(0.23,1,0.32,1) 0.1s both;
    }
    .up-eyebrow {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 5px 14px 5px 8px; border-radius: 100px;
      background: rgba(201,168,76,0.08);
      border: 1px solid rgba(201,168,76,0.22);
      font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
      color: var(--gold); text-transform: uppercase; margin-bottom: 22px;
    }
    .up-eyebrow-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--gold); }
    .up-h1 {
      font-family: var(--font-serif); font-size: clamp(2.4rem, 5vw, 3.8rem);
      font-weight: 400; line-height: 1.08; color: var(--white);
      margin-bottom: 18px; letter-spacing: -0.01em;
    }
    .up-h1 em { font-style: italic; color: var(--gold); }
    .up-sub {
      font-size: 0.98rem; color: var(--text); font-weight: 300;
      line-height: 1.75; max-width: 420px; margin: 0 auto;
    }

    @keyframes riseIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* ── Upload zone ── */
    .up-zone {
      border: 1.5px dashed rgba(201,168,76,0.28);
      border-radius: var(--radius-xl); padding: 56px 32px;
      text-align: center; cursor: pointer;
      transition: all 0.35s cubic-bezier(0.23, 1, 0.32, 1);
      background: var(--glass-bg); backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
      position: relative; overflow: hidden;
      animation: riseIn 0.8s cubic-bezier(0.23,1,0.32,1) 0.25s both;
    }
    .up-zone::before {
      content: ''; position: absolute; inset: 0;
      background: radial-gradient(circle at 50% 50%, rgba(201,168,76,0.04), transparent 70%);
      pointer-events: none;
    }
    .up-zone:hover, .up-zone.dragging {
      border-color: rgba(201,168,76,0.55);
      box-shadow: 0 0 0 1px rgba(201,168,76,0.1), 0 24px 60px rgba(0,0,0,0.5);
      transform: translateY(-3px);
    }
    .up-zone-icon {
      width: 72px; height: 72px; border-radius: 20px;
      background: rgba(201,168,76,0.09); border: 1px solid var(--glass-border);
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 24px; transition: all 0.3s;
    }
    .up-zone:hover .up-zone-icon, .up-zone.dragging .up-zone-icon {
      background: rgba(201,168,76,0.16); border-color: rgba(201,168,76,0.45);
    }
    .up-zone-title { font-family: var(--font-serif); font-size: 1.65rem; color: var(--white); margin-bottom: 8px; }
    .up-zone-sub { font-size: 0.88rem; color: var(--text-dim); margin-bottom: 28px; font-weight: 300; }

    /* Buttons */
    .btn-teal {
      font-family: var(--font-sans); font-size: 0.84rem; font-weight: 500;
      padding: 11px 28px; border-radius: 100px;
      background: linear-gradient(135deg, rgba(201,168,76,0.92) 0%, rgba(170,130,40,0.92) 100%);
      border: 1px solid rgba(240,210,130,0.3); color: #0a0f1a; cursor: pointer; text-decoration: none;
      display: inline-flex; align-items: center; gap: 8px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.2), 0 6px 20px rgba(201,168,76,0.22);
      transition: all 0.3s cubic-bezier(0.23,1,0.32,1);
    }
    .btn-teal:hover { transform: translateY(-2px); box-shadow: inset 0 1px 0 rgba(255,255,255,0.28), 0 10px 32px rgba(201,168,76,0.35); }

    .btn-glass {
      font-family: var(--font-sans); font-size: 0.84rem; font-weight: 500;
      padding: 10px 24px; border-radius: 100px;
      background: var(--glass-bg); border: 1px solid var(--glass-border);
      color: var(--white); cursor: pointer; text-decoration: none;
      display: inline-flex; align-items: center; gap: 8px;
      backdrop-filter: blur(16px);
      box-shadow: inset 0 1px 0 var(--glass-shine);
      transition: all 0.3s cubic-bezier(0.23,1,0.32,1);
    }
    .btn-glass:hover { border-color: rgba(201,168,76,0.45); transform: translateY(-1px); }

    /* ── File pill ── */
    .file-pill {
      display: flex; align-items: center; gap: 14px;
      background: var(--glass-bg); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--glass-border); border-radius: 14px; padding: 14px 18px; margin-bottom: 28px;
      animation: riseIn 0.5s cubic-bezier(0.23,1,0.32,1) both;
    }
    .file-pill-icon {
      width: 42px; height: 42px; border-radius: 10px;
      background: rgba(201,168,76,0.1); border: 1px solid var(--glass-border);
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .file-pill-name { font-size: 0.9rem; font-weight: 500; color: var(--white); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .file-pill-size { font-size: 0.75rem; color: var(--text-dim); margin-top: 3px; }
    .file-pill-close {
      margin-left: auto; width: 30px; height: 30px; border-radius: 8px;
      background: transparent; border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      color: var(--text-dim); transition: all 0.2s; flex-shrink: 0;
    }
    .file-pill-close:hover { background: rgba(220,38,38,0.12); color: #fca5a5; }

    /* ── Options ── */
    .options-section { margin-bottom: 28px; animation: riseIn 0.6s cubic-bezier(0.23,1,0.32,1) 0.1s both; }
    .options-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; color: var(--gold-dim); margin-bottom: 14px; }
    .options-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }

    .option-card {
      position: relative; background: rgba(10, 18, 30, 0.55);
      border: 1px solid var(--glass-border); border-radius: 14px;
      padding: 14px 16px; cursor: pointer;
      transition: all 0.25s cubic-bezier(0.23, 1, 0.32, 1);
      user-select: none; overflow: hidden;
    }
    .option-card::before {
      content: ''; position: absolute; top: 0; left: 0;
      width: 3px; height: 100%; background: var(--gold);
      transform: scaleY(0); transition: transform 0.25s cubic-bezier(0.23, 1, 0.32, 1);
      border-radius: 0 2px 2px 0;
    }
    .option-card:hover { border-color: rgba(201,168,76,0.32); transform: translateY(-2px); }
    .option-card.active { border-color: rgba(201,168,76,0.38); background: rgba(201,168,76,0.06); }
    .option-card.active::before { transform: scaleY(1); }
    .option-card-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; }
    .option-card-left { display: flex; align-items: center; gap: 10px; }
    .option-icon {
      width: 30px; height: 30px; border-radius: 8px;
      background: rgba(201,168,76,0.08);
      display: flex; align-items: center; justify-content: center; transition: background 0.2s;
    }
    .option-card.active .option-icon { background: rgba(201,168,76,0.16); }
    .option-name { font-size: 0.88rem; font-weight: 500; color: var(--white); }
    .option-toggle {
      width: 34px; height: 18px; border-radius: 9px;
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1);
      position: relative; transition: all 0.25s; flex-shrink: 0;
    }
    .option-toggle::after {
      content: ''; position: absolute; left: 2px; top: 50%; transform: translateY(-50%);
      width: 12px; height: 12px; border-radius: 50%;
      background: var(--text-dim); transition: all 0.25s cubic-bezier(0.23, 1, 0.32, 1);
    }
    .option-card.active .option-toggle { background: var(--gold); border-color: var(--gold); }
    .option-card.active .option-toggle::after { left: calc(100% - 14px); background: #0a0f1a; }
    .option-desc { font-size: 0.76rem; color: var(--text-dim); line-height: 1.5; padding-left: 40px; }

    /* ── Customize ── */
    .customize-section {
      background: var(--glass-bg); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--glass-border); border-radius: var(--radius-lg);
      padding: 18px 22px; margin-bottom: 16px;
      animation: riseIn 0.45s cubic-bezier(0.23,1,0.32,1) both;
    }
    .customize-eyebrow {
      font-size: 0.6rem; font-weight: 500; letter-spacing: 0.22em;
      text-transform: uppercase; color: var(--gold-dim);
      margin-bottom: 14px; display: inline-flex; align-items: center; gap: 8px;
    }
    .customize-eyebrow::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--gold-dim); }
    .customize-row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 14px; padding: 10px 0; border-bottom: 1px solid rgba(201,168,76,0.07);
    }
    .customize-row:last-child { border-bottom: none; padding-bottom: 0; }
    .customize-row:first-of-type { padding-top: 0; }
    .customize-label { font-size: 0.83rem; color: var(--text); font-weight: 400; display: flex; flex-direction: column; gap: 3px; }
    .customize-label-hint { font-size: 0.72rem; color: var(--text-dim); font-weight: 300; }

    .seg-toggle { display: inline-flex; padding: 3px; background: rgba(5, 10, 20, 0.6); border: 1px solid var(--glass-border); border-radius: 100px; gap: 2px; }
    .seg-btn {
      font-family: var(--font-sans); font-size: 0.78rem; font-weight: 500;
      padding: 6px 16px; border-radius: 100px; border: none;
      background: transparent; color: var(--text-dim); cursor: pointer;
      transition: all 0.22s ease; letter-spacing: 0.02em;
    }
    .seg-btn:hover { color: var(--text); }
    .seg-btn.active {
      background: linear-gradient(135deg, rgba(201,168,76,0.88) 0%, rgba(170,130,40,0.88) 100%);
      color: #0a0f1a; box-shadow: 0 2px 10px rgba(201,168,76,0.2);
    }

    /* ── Topic field ── */
    .topic-wrap {
      background: var(--glass-bg); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--glass-border); border-radius: 14px;
      padding: 16px 20px; display: flex; align-items: center; gap: 14px;
      margin-bottom: 24px; animation: riseIn 0.5s cubic-bezier(0.23,1,0.32,1) both;
    }
    .topic-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: var(--gold-dim); white-space: nowrap; flex-shrink: 0; }
    .topic-input { flex: 1; background: transparent; border: none; outline: none; color: var(--white); font-family: var(--font-sans); font-size: 0.9rem; caret-color: var(--gold); }
    .topic-input::placeholder { color: var(--text-dim); }

    /* ── Error bar ── */
    .error-bar {
      background: rgba(220,38,38,0.07); border: 1px solid rgba(220,38,38,0.22);
      border-radius: 12px; padding: 12px 18px;
      display: flex; align-items: center; gap: 10px;
      color: #fca5a5; font-size: 0.88rem; margin-bottom: 20px;
      animation: riseIn 0.4s ease both;
    }

    .actions { display: flex; justify-content: center; margin-top: 8px; }

    /* ── Processing ── */
    .phase-card {
      background: var(--glass-bg); backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
      border: 1px solid var(--glass-border); border-radius: var(--radius-xl);
      padding: 28px 32px; margin-bottom: 20px;
      box-shadow: inset 0 1px 0 var(--glass-shine);
      animation: riseIn 0.6s cubic-bezier(0.23,1,0.32,1) both;
    }
    .phase-card-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--glass-border);
    }
    .phase-filename { font-size: 0.92rem; font-weight: 500; color: var(--white); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 380px; }
    .phase-hint { font-size: 0.74rem; color: var(--text-dim); margin-top: 3px; font-weight: 300; }
    .live-pill { display: flex; align-items: center; gap: 6px; background: rgba(201,168,76,0.07); border: 1px solid rgba(201,168,76,0.22); border-radius: 100px; padding: 4px 12px; flex-shrink: 0; }
    .live-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; animation: pulseDot 1.4s ease-in-out infinite; }
    @keyframes pulseDot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }
    .live-label { font-size: 0.62rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: var(--gold-dim); }
    .phases-list { display: flex; flex-direction: column; gap: 14px; }
    .phase-row { display: flex; align-items: flex-start; gap: 14px; transition: opacity 0.5s ease; }
    .phase-row.dim { opacity: 0.25; }
    .phase-status { width: 24px; height: 24px; flex-shrink: 0; margin-top: 1px; display: flex; align-items: center; justify-content: center; }
    .status-done { width: 24px; height: 24px; border-radius: 50%; background: rgba(34,197,94,0.2); border: 1px solid rgba(34,197,94,0.4); display: flex; align-items: center; justify-content: center; }
    .status-spin { width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid var(--glass-border); border-top-color: var(--gold); animation: spinAnim 0.75s linear infinite; }
    @keyframes spinAnim { to { transform: rotate(360deg); } }
    .status-idle { width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid rgba(255,255,255,0.1); }
    .phase-text { flex: 1; }
    .phase-name { font-size: 0.88rem; font-weight: 500; letter-spacing: 0.02em; line-height: 1.3; }
    .phase-name.done { color: #86efac; }
    .phase-name.active { color: var(--white); }
    .phase-name.idle { color: var(--text-dim); }
    .phase-msg { font-size: 0.75rem; color: var(--gold-dim); font-style: italic; margin-top: 3px; animation: fadeUp 0.4s ease both; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }

    /* ── Skeleton ── */
    .skeleton-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
    .skeleton-card {
      background: var(--glass-bg); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--glass-border); border-radius: var(--radius-lg); padding: 20px; overflow: hidden;
    }
    .skel {
      background: linear-gradient(90deg, rgba(201,168,76,0.06) 25%, rgba(201,168,76,0.12) 50%, rgba(201,168,76,0.06) 75%);
      background-size: 200% 100%; animation: shimmer 2s infinite; border-radius: 6px;
    }
    @keyframes shimmer { to { background-position: -200% center; } }

    /* Results actions */
    .results-actions {
      display: flex; justify-content: center; gap: 14px; flex-wrap: wrap;
      margin-top: 24px;
      animation: riseIn 0.6s cubic-bezier(0.23,1,0.32,1) 0.3s both;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.22); border-radius: 3px; }
  `}</style>
);

/* ─── Phase definitions ──────────────────────────────────────────────────────── */
const getPhases = (options) => {
  const p = [{ id: 'init', label: 'Initialising pipeline', keywords: ['initializ', 'starting'] }];
  if (options.transcript) p.push({ id: 'transcription', label: 'Transcribing audio', keywords: ['transcrib', 'audio', 'speech', 'whisper', 'analyzing audio', 'loading transcription', 'processing speech', 'formatting transcription'] });
  if (options.keyframes) p.push({ id: 'keyframes', label: 'Extracting keyframes', keywords: ['keyframe', 'frame', 'visual', 'cluster'] });
  if (options.summary || options.topicBased) p.push({ id: 'summary', label: 'Generating summary', keywords: ['summar', 'gemini', 'topic', 'ai summary', 'generating ai', 'summary complete', 'generating summary'] });
  if (options.rag) p.push({ id: 'rag', label: 'Indexing knowledge', keywords: ['rag', 'chunk', 'graph', 'vector', 'index', 'retrieval', 'semantic'] });
  if (options.subtitles) p.push({ id: 'subtitles', label: 'Generating subtitles', keywords: ['subtitle', 'srt', 'caption', 'sync'] });
  return p;
};

const PHASE_MESSAGES = {
  init: ['Warming up the engines…', 'Getting everything ready…', 'Firing up the pipeline…'],
  transcription: ['Listening to every word…', 'Whispering with Whisper…', 'Decoding the audio waves…', 'Converting speech to knowledge…'],
  keyframes: ['Picking the perfect moments…', 'Clustering the visual story…', 'Finding frames that matter…'],
  summary: ['Reading between the lines…', 'Asking Gemini nicely…', 'Distilling the essence…', 'Crafting the summary…'],
  rag: ['Building the knowledge graph…', 'Weaving the semantic web…', 'Thinking in vectors…'],
  subtitles: ['Syncing words to timestamps…', 'Writing the captions…', 'Timestamping every syllable…'],
  default: ['Working hard…', 'Almost there…', 'The gears are turning…'],
};

const OPTION_META = [
  { key: 'transcript', icon: FileText,  label: 'Transcript',      desc: 'Full text transcription' },
  { key: 'summary',    icon: Brain,     label: 'General Summary', desc: 'Hybrid retrieval + Gemini' },
  { key: 'topicBased', icon: Languages, label: 'Topic Summary',   desc: 'Focus on a specific topic' },
  { key: 'subtitles',  icon: Clock,     label: 'Subtitles',       desc: 'Time-synced SRT / WebVTT' },
];

/* ── Skeleton ── */
const SkeletonCard = ({ lines = 6 }) => (
  <div className="skeleton-card">
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
      <div className="skel" style={{ width: 26, height: 26, borderRadius: 8 }} />
      <div className="skel" style={{ width: 110, height: 13 }} />
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skel" style={{ height: 10, width: i % 3 === 2 ? '65%' : i % 3 === 1 ? '82%' : '100%' }} />
      ))}
    </div>
  </div>
);

/* ─── Main ──────────────────────────────────────────────────────────────────── */
const VideoUpload = () => {
  const [videoFile, setVideoFile]           = useState(null);
  const [isDragging, setIsDragging]         = useState(false);
  const [isProcessing, setIsProcessing]     = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const [results, setResults]               = useState(null);
  const [selectedOptions, setSelectedOptions] = useState({
    subtitles: true, transcript: true, summary: true,
    keyframes: true, rag: true, topicBased: false,
  });
  const [selectedTopic, setSelectedTopic]   = useState('');
  const [summaryLength, setSummaryLength]   = useState('medium');
  const [subtitleFormat, setSubtitleFormat] = useState('srt');
  const [error, setError]                   = useState(null);

  const [phaseMessageIdx, setPhaseMessageIdx]     = useState(0);
  const [currentPhaseId, setCurrentPhaseId]       = useState('init');
  const [completedPhaseIds, setCompletedPhaseIds] = useState([]);
  const [showRawSrt, setShowRawSrt]               = useState(false);

  useEffect(() => {
    if (!isProcessing) { setPhaseMessageIdx(0); setCurrentPhaseId('init'); setCompletedPhaseIds([]); return; }
    const t = setInterval(() => setPhaseMessageIdx(i => i + 1), 4000);
    return () => clearInterval(t);
  }, [isProcessing]);

  useEffect(() => {
    if (!processingStep || !isProcessing) return;
    const step = processingStep.toLowerCase();
    const phases = getPhases(selectedOptions);
    for (let i = 0; i < phases.length; i++) {
      if (phases[i].keywords.some(k => step.includes(k))) {
        if (phases[i].id !== currentPhaseId) {
          setCurrentPhaseId(phases[i].id);
          setCompletedPhaseIds(phases.slice(0, i).map(p => p.id));
          setPhaseMessageIdx(0);
        }
        return;
      }
    }
  }, [processingStep]); // eslint-disable-line

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file?.type.startsWith('video/')) { setVideoFile(file); setResults(null); }
  };

  const handleDrop = (e) => {
    e.preventDefault(); setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.type.startsWith('video/')) { setVideoFile(file); setResults(null); }
  };

  const toggleOption = (key) => setSelectedOptions(prev => ({ ...prev, [key]: !prev[key] }));

  const handleProcess = async () => {
    if (!videoFile) return;
    setIsProcessing(true); setResults(null); setError(null);
    setCurrentPhaseId('init'); setCompletedPhaseIds([]); setPhaseMessageIdx(0);
    try {
      const options = {
        transcript: selectedOptions.transcript, summary: selectedOptions.summary,
        subtitles: selectedOptions.subtitles,
        keyframes: true,   // always on — Phase 1 of pipeline
        rag: true,         // always on — core indexing
        topic_based: selectedOptions.topicBased,
        topic: selectedOptions.topicBased ? selectedTopic : null,
        download_format: subtitleFormat, summary_length: summaryLength, use_gpu: true,
      };
      const response = await apiService.uploadVideo(videoFile, options);
      apiService.pollStatus(response.task_id, (status) => {
        setProcessingStep(status.current_step || '');
        if (status.status === 'completed') {
          const rawSummary = status.results?.summary;
          const rawTopicSummary = status.results?.topic_summary;
          setResults({
            transcript: status.results?.transcript?.segments
              ? status.results.transcript.segments.map(s => s.text).join(' ')
              : status.results?.transcript || 'No transcript available',
            // Keep timed segments for the in-app caption overlay (karaoke / flashy styles).
            segments: status.results?.transcript?.segments || [],
            summary: {
              text: rawSummary?.summary || (typeof rawSummary === 'string' ? rawSummary : 'No summary available'),
              key_points: rawSummary?.key_points || [],
              summary_type: rawSummary?.summary_type || 'extractive',
              method: rawSummary?.method,
              summary_length: rawSummary?.summary_length,
              retrieved_chunk_count: rawSummary?.retrieved_chunk_count,
            },
            topic_summary: rawTopicSummary ? {
              text: rawTopicSummary?.summary || '',
              key_points: rawTopicSummary?.key_points || [],
              summary_type: rawTopicSummary?.summary_type || 'gemini',
              topic: rawTopicSummary?.topic,
              retrieved_chunk_count: rawTopicSummary?.retrieved_chunk_count,
            } : null,
            subtitles: status.results?.subtitles?.subtitles || status.results?.subtitles || 'No subtitles available',
            keyframes: status.results?.keyframes?.keyframes
              ? status.results.keyframes.keyframes.map(kf => ({ ...kf, path: kf.frame_path || kf.path || 'unknown' }))
              : [],
            rag: status.results?.rag || null,
            citations: status.results?.citations || [],
            video_url: status.results?.video_url || null,
            subtitles_vtt_url: status.results?.subtitles_vtt_url || null,
            task_id: status.results?.task_id || response.task_id,
          });
          setIsProcessing(false);
        } else if (status.status === 'failed' || status.status === 'error') {
          setError(status.error || 'Processing failed');
          setIsProcessing(false);
        }
      });
    } catch (err) {
      setError(err.message || 'Failed to start processing');
      setIsProcessing(false);
    }
  };

  const handleDownload = (type) => {
    let content = '', filename = '';
    switch (type) {
      case 'transcript': content = typeof results.transcript === 'string' ? results.transcript : ''; filename = `transcript_${videoFile.name.replace(/\.[^/.]+$/, '')}.txt`; break;
      case 'subtitles': content = typeof results.subtitles === 'string' ? results.subtitles : ''; filename = `subtitles_${videoFile.name.replace(/\.[^/.]+$/, '')}.${subtitleFormat}`; break;
      case 'summary':
        content = results.summary?.text || '';
        if (results.summary?.key_points?.length) content += '\n\nKey Points:\n' + results.summary.key_points.map(p => `• ${p}`).join('\n');
        filename = `summary_${videoFile.name.replace(/\.[^/.]+$/, '')}.txt`; break;
      case 'keyframes':
        content = (Array.isArray(results.keyframes) ? results.keyframes : []).map(kf => `${kf.timestamp || kf.time}: ${kf.description || 'Keyframe'}`).join('\n');
        filename = `keyframes_${videoFile.name.replace(/\.[^/.]+$/, '')}.txt`; break;
      case 'rag': content = results.rag?.summary?.summary || ''; filename = `rag_summary_${videoFile.name.replace(/\.[^/.]+$/, '')}.txt`; break;
    }
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  };

  const resetUpload = () => {
    setVideoFile(null); setResults(null); setIsProcessing(false); setError(null);
    setSelectedTopic(''); setSummaryLength('medium'); setSubtitleFormat('srt');
    setSelectedOptions({ subtitles: true, transcript: true, summary: true, keyframes: true, rag: true, topicBased: false });
  };

  const activePhases = getPhases(selectedOptions);

  return (
    <div className="up-wrap">
      <PageStyles />
      <div className="up-mesh" />
      <div className="up-orb up-orb-1" />
      <div className="up-orb up-orb-2" />

      <nav className="up-nav">
        <Link to="/landing" className="up-nav-brand">
          <img src="/logo.png" alt="Mimir" />
          <span className="up-nav-brand-name">Mimir</span>
        </Link>
        <div className="up-nav-links">
          <Link to="/profile" className="up-nav-a">Profile</Link>
          <Link to="/landing" className="up-nav-a">Home</Link>
        </div>
      </nav>

      <div className="up-page">
        <div className="up-header">
          <div className="up-eyebrow">
            <div className="up-eyebrow-dot" />
            Video Processing Studio
          </div>
          <h1 className="up-h1">
            Turn video into<br /><em>structured knowledge.</em>
          </h1>
          <p className="up-sub">
            Upload your video, choose your outputs, and let Mimir do the rest.
          </p>
        </div>

        {/* Upload zone */}
        {!videoFile && (
          <div
            className={`up-zone${isDragging ? ' dragging' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="up-zone-icon">
              <Upload size={26} color={isDragging ? '#c9a84c' : 'rgba(201,168,76,0.65)'} />
            </div>
            <h3 className="up-zone-title">{isDragging ? 'Drop it here' : 'Drop your video'}</h3>
            <p className="up-zone-sub">MP4 · MKV · MOV · AVI — drag and drop or click to browse</p>
            <input type="file" accept="video/*" onChange={handleFileUpload} id="video-upload" style={{ display: 'none' }} />
            <label htmlFor="video-upload">
              <span className="btn-teal" style={{ cursor: 'pointer' }}>
                <FileVideo size={15} />
                Choose file
              </span>
            </label>
          </div>
        )}

        {/* File selected — config */}
        {videoFile && !results && (
          <>
            {!isProcessing && (
              <div>
                <div className="file-pill">
                  <div className="file-pill-icon">
                    <FileVideo size={18} color="rgba(201,168,76,0.85)" />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="file-pill-name">{videoFile.name}</div>
                    <div className="file-pill-size">{(videoFile.size / 1024 / 1024).toFixed(2)} MB</div>
                  </div>
                  <button onClick={resetUpload} className="file-pill-close"><X size={15} /></button>
                </div>

                <div className="options-section">
                  <p className="options-label">Select outputs</p>
                  <div className="options-grid">
                    {OPTION_META.map(({ key, icon: Icon, label, desc }) => (
                      <div key={key} className={`option-card${selectedOptions[key] ? ' active' : ''}`} onClick={() => toggleOption(key)}>
                        <div className="option-card-top">
                          <div className="option-card-left">
                            <div className="option-icon">
                              <Icon size={14} color={selectedOptions[key] ? 'rgba(201,168,76,0.95)' : 'rgba(201,168,76,0.4)'} />
                            </div>
                            <span className="option-name">{label}</span>
                          </div>
                          <div className="option-toggle" />
                        </div>
                        <p className="option-desc">{desc}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {(selectedOptions.summary || selectedOptions.topicBased || selectedOptions.subtitles) && (
                  <div className="customize-section">
                    <p className="customize-eyebrow">Customize</p>
                    {(selectedOptions.summary || selectedOptions.topicBased) && (
                      <div className="customize-row">
                        <div className="customize-label">
                          Summary length
                          <span className="customize-label-hint">
                            {summaryLength === 'short' && 'Brief — 1–2 paragraphs, 3 key points'}
                            {summaryLength === 'medium' && 'Balanced — 3–5 paragraphs, 5 key points'}
                            {summaryLength === 'long' && 'Detailed — 6–8 paragraphs, 8 key points'}
                          </span>
                        </div>
                        <div className="seg-toggle">
                          {['short', 'medium', 'long'].map((l) => (
                            <button key={l} type="button" onClick={() => setSummaryLength(l)} className={`seg-btn${summaryLength === l ? ' active' : ''}`}>
                              {l[0].toUpperCase() + l.slice(1)}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {selectedOptions.subtitles && (
                      <div className="customize-row">
                        <div className="customize-label">
                          Subtitle format
                          <span className="customize-label-hint">
                            {subtitleFormat === 'srt' ? 'SRT — universal video player support' : 'WebVTT — modern web standard'}
                          </span>
                        </div>
                        <div className="seg-toggle">
                          {['srt', 'vtt'].map((f) => (
                            <button key={f} type="button" onClick={() => setSubtitleFormat(f)} className={`seg-btn${subtitleFormat === f ? ' active' : ''}`}>
                              {f.toUpperCase()}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {selectedOptions.topicBased && (
                  <div className="topic-wrap">
                    <span className="topic-label">Topic</span>
                    <input type="text" value={selectedTopic} onChange={(e) => setSelectedTopic(e.target.value)}
                      placeholder="e.g. machine learning, business strategy…" className="topic-input" />
                  </div>
                )}

                {error && (
                  <div className="error-bar">
                    <AlertCircle size={15} color="#fca5a5" /><span>{error}</span>
                  </div>
                )}

                <div className="actions">
                  <button onClick={handleProcess} className="btn-teal" style={{ fontSize: '0.9rem', padding: '13px 36px' }}>
                    <Zap size={15} />Start processing
                  </button>
                </div>
              </div>
            )}

            {/* Processing */}
            {isProcessing && (
              <div>
                <div className="phase-card">
                  <div className="phase-card-header">
                    <div>
                      <div className="phase-filename">{videoFile.name}</div>
                      <div className="phase-hint">Processing in progress — this may take a few minutes</div>
                    </div>
                    <div className="live-pill">
                      <div className="live-dot" />
                      <span className="live-label">Live</span>
                    </div>
                  </div>
                  <div className="phases-list">
                    {activePhases.map((phase) => {
                      const isDone   = completedPhaseIds.includes(phase.id);
                      const isActive = currentPhaseId === phase.id;
                      const msgs     = PHASE_MESSAGES[phase.id] || PHASE_MESSAGES.default;
                      const msg      = msgs[phaseMessageIdx % msgs.length];
                      return (
                        <div key={phase.id} className={`phase-row${!isDone && !isActive ? ' dim' : ''}`}>
                          <div className="phase-status">
                            {isDone ? (
                              <div className="status-done">
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                  <polyline points="20 6 9 17 4 12" />
                                </svg>
                              </div>
                            ) : isActive ? (
                              <div className="status-spin" />
                            ) : (
                              <div className="status-idle" />
                            )}
                          </div>
                          <div className="phase-text">
                            <p className={`phase-name ${isDone ? 'done' : isActive ? 'active' : 'idle'}`}>{phase.label}</p>
                            {isActive && <p className="phase-msg" key={msg}>{msg}</p>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="skeleton-grid">
                  {selectedOptions.transcript && <SkeletonCard lines={8} />}
                  {selectedOptions.subtitles && <SkeletonCard lines={7} />}
                  {(selectedOptions.summary || selectedOptions.topicBased) && <SkeletonCard lines={5} />}
                  {selectedOptions.keyframes && <SkeletonCard lines={4} />}
                  {selectedOptions.rag && <SkeletonCard lines={5} />}
                </div>
              </div>
            )}
          </>
        )}

        {/* Results */}
        {results && (
          <>
            <ResultsPanel
              results={results}
              selectedOptions={selectedOptions}
              subtitleFormat={subtitleFormat}
              videoFile={videoFile}
              onDownload={handleDownload}
              showRawSrt={showRawSrt}
              setShowRawSrt={setShowRawSrt}
              taskId={results.task_id}
            />
            <div className="results-actions">
              <button onClick={resetUpload} className="btn-teal">
                <RotateCcw size={14} />Process another
              </button>
              {results.task_id && (
                <Link to={`/logs/${results.task_id}`} className="btn-glass">
                  Pipeline Logs<ChevronRight size={14} />
                </Link>
              )}
              <Link to="/profile" className="btn-glass">
                View history<ChevronRight size={14} />
              </Link>
            </div>
          </>
        )}

      </div>
    </div>
  );
};

export default VideoUpload;
