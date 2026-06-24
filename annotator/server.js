'use strict';

const express = require('express');
const path    = require('path');
const fs      = require('fs');
const os      = require('os');
const { spawn } = require('child_process');

const app  = express();
const PORT = process.env.PORT || 3000;

const VIDEOS_DIR   = path.resolve(__dirname, '..', 'videos');
const SESSIONS_DIR = path.resolve(__dirname, 'sessions');

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Single active session — sufficient for LAN use
let session = null;

// ── Videos ──────────────────────────────────────────────────────────────────

app.get('/api/videos', (_req, res) => {
  try {
    const files = fs.readdirSync(VIDEOS_DIR)
      .filter(f => /\.(mp4|mov|avi|mkv|webm)$/i.test(f))
      .sort();
    res.json(files);
  } catch {
    res.json([]);
  }
});

// ── Extract frames via ffmpeg (SSE) ─────────────────────────────────────────

app.get('/api/extract', (req, res) => {
  const { video, fps } = req.query;
  if (!video || !fps) return res.status(400).json({ error: 'Parâmetros ausentes' });

  const videoPath = path.join(VIDEOS_DIR, path.basename(video));
  if (!fs.existsSync(videoPath)) return res.status(404).json({ error: 'Vídeo não encontrado' });

  const sessionName = path.parse(video).name.replace(/[^a-z0-9_-]/gi, '_');
  const sessionDir  = path.join(SESSIONS_DIR, sessionName);
  const framesDir   = path.join(sessionDir, 'frames');
  fs.mkdirSync(framesDir, { recursive: true });

  // Apaga frames anteriores da sessão
  fs.readdirSync(framesDir).forEach(f => fs.unlinkSync(path.join(framesDir, f)));

  res.writeHead(200, {
    'Content-Type':  'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection':    'keep-alive',
  });

  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

  const ffmpeg = spawn('ffmpeg', [
    '-i', videoPath,
    '-vf', `fps=${fps}`,
    '-start_number', '0',
    '-q:v', '2',
    path.join(framesDir, '%04d.jpg'),
    '-progress', 'pipe:1',
    '-y', '-nostats',
  ]);

  let estimated = 0;

  ffmpeg.stderr.on('data', chunk => {
    const m = chunk.toString().match(/Duration:\s*(\d+):(\d+):([\d.]+)/);
    if (m) {
      const secs = parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseFloat(m[3]);
      estimated = Math.ceil(secs * parseFloat(fps));
      send({ total: estimated });
    }
  });

  ffmpeg.stdout.on('data', chunk => {
    const m = chunk.toString().match(/frame=\s*(\d+)/);
    if (m) send({ frame: parseInt(m[1]), total: estimated });
  });

  ffmpeg.on('close', code => {
    if (code !== 0) {
      send({ error: `ffmpeg encerrou com código ${code}` });
      return res.end();
    }

    const total = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).length;
    const annotFile = path.join(sessionDir, 'annotations.json');
    let annotations = new Array(total).fill(null);
    if (fs.existsSync(annotFile)) {
      try {
        const saved = JSON.parse(fs.readFileSync(annotFile, 'utf8'));
        if (saved.length === total) annotations = saved;
      } catch {}
    }

    const sessionConfig = { name: sessionName, fps: parseFloat(fps), total };
    fs.writeFileSync(path.join(sessionDir, 'session.json'), JSON.stringify(sessionConfig));

    session = { name: sessionName, sessionDir, framesDir, annotFile, total, annotations, fps: parseFloat(fps) };
    send({ done: true, total });
    res.end();
  });

  req.on('close', () => { try { ffmpeg.kill(); } catch {} });
});

// ── Serve frame image ────────────────────────────────────────────────────────

app.get('/api/frame/:index', (req, res) => {
  if (!session) return res.status(400).end();
  const idx = parseInt(req.params.index);
  if (isNaN(idx) || idx < 0 || idx >= session.total) return res.status(404).end();
  const p = path.join(session.framesDir, String(idx).padStart(4, '0') + '.jpg');
  if (!fs.existsSync(p)) return res.status(404).end();
  res.setHeader('Cache-Control', 'public, max-age=86400');
  res.sendFile(p);
});

// ── Session state ────────────────────────────────────────────────────────────

app.get('/api/state', (_req, res) => {
  if (!session) return res.json({ loaded: false });
  res.json({
    loaded:      true,
    name:        session.name,
    total:       session.total,
    fps:         session.fps,
    annotations: session.annotations,
  });
});

// ── Save annotation ──────────────────────────────────────────────────────────

app.post('/api/annotate/:index', (req, res) => {
  if (!session) return res.status(400).end();
  const idx = parseInt(req.params.index);
  if (isNaN(idx) || idx < 0 || idx >= session.total) return res.status(400).end();

  const { visibility, x, y } = req.body;
  if (visibility === null || visibility === undefined) {
    session.annotations[idx] = null;
  } else {
    session.annotations[idx] = { visibility: Number(visibility), x: x ?? null, y: y ?? null };
  }

  fs.writeFileSync(session.annotFile, JSON.stringify(session.annotations));
  res.json({ ok: true });
});

// ── Export CSV ───────────────────────────────────────────────────────────────

app.get('/api/export', (_req, res) => {
  if (!session) return res.status(400).end();
  const rows = ['file name,visibility,x-coordinate,y-coordinate,status'];
  session.annotations.forEach((ann, i) => {
    const name = String(i).padStart(4, '0') + '.jpg';
    if (ann?.visibility === 1) {
      rows.push(`${name},1,${ann.x},${ann.y},0`);
    } else {
      rows.push(`${name},0,,,0`);
    }
  });
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', 'attachment; filename="Label.csv"');
  res.end(rows.join('\n'));
});

// ── Sessions list & load ─────────────────────────────────────────────────────

app.get('/api/sessions', (_req, res) => {
  try {
    const dirs = fs.readdirSync(SESSIONS_DIR)
      .filter(f => fs.statSync(path.join(SESSIONS_DIR, f)).isDirectory())
      .sort();
    res.json(dirs);
  } catch { res.json([]); }
});

app.post('/api/session/load', (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).end();

  const sessionDir  = path.join(SESSIONS_DIR, name);
  const framesDir   = path.join(sessionDir, 'frames');
  const annotFile   = path.join(sessionDir, 'annotations.json');
  if (!fs.existsSync(framesDir)) return res.status(404).end();

  const total = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).length;
  let annotations = new Array(total).fill(null);
  if (fs.existsSync(annotFile)) {
    try {
      const saved = JSON.parse(fs.readFileSync(annotFile, 'utf8'));
      if (saved.length === total) annotations = saved;
    } catch {}
  }

  let fps = 30;
  const configFile = path.join(sessionDir, 'session.json');
  if (fs.existsSync(configFile)) {
    try { fps = JSON.parse(fs.readFileSync(configFile, 'utf8')).fps || 30; } catch {}
  }

  session = { name, sessionDir, framesDir, annotFile, total, annotations, fps };
  res.json({ ok: true, total });
});

// ── Fill absent ──────────────────────────────────────────────────────────────

app.post('/api/fill-absent', (_req, res) => {
  if (!session) return res.status(400).end();
  let filled = 0;
  session.annotations = session.annotations.map(ann => {
    if (ann !== null) return ann;
    filled++;
    return { visibility: 0, x: null, y: null };
  });
  fs.writeFileSync(session.annotFile, JSON.stringify(session.annotations));
  res.json({ filled, annotations: session.annotations });
});

// ── Crop CSV ─────────────────────────────────────────────────────────────────

app.get('/api/crop/csv', (req, res) => {
  if (!session) return res.status(400).end();
  const start = Math.max(0, parseInt(req.query.start) || 0);
  const end   = Math.min(parseInt(req.query.end ?? session.total - 1), session.total - 1);
  if (start > end) return res.status(400).json({ error: 'start > end' });

  const rows = ['file name,visibility,x-coordinate,y-coordinate,status'];
  for (let i = start; i <= end; i++) {
    const name = String(i - start).padStart(4, '0') + '.jpg';
    const ann  = session.annotations[i];
    if (ann?.visibility === 1) rows.push(`${name},1,${ann.x},${ann.y},0`);
    else rows.push(`${name},0,,,0`);
  }

  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="Label_${start}-${end}.csv"`);
  res.end(rows.join('\n'));
});

// ── Crop video (SSE) ──────────────────────────────────────────────────────────

app.get('/api/crop/video', (req, res) => {
  if (!session) return res.status(400).end();
  const start = Math.max(0, parseInt(req.query.start) || 0);
  const end   = Math.min(parseInt(req.query.end ?? session.total - 1), session.total - 1);
  if (start > end) return res.status(400).end();

  const count    = end - start + 1;
  const fps      = session.fps || 30;
  const filename = `crop_${start}_${end}.mp4`;
  const outPath  = path.join(session.sessionDir, filename);

  res.writeHead(200, {
    'Content-Type':  'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection':    'keep-alive',
  });
  const send = obj => res.write(`data: ${JSON.stringify(obj)}\n\n`);
  send({ total: count });

  const ffmpeg = spawn('ffmpeg', [
    '-framerate', String(fps),
    '-start_number', String(start),
    '-i', path.join(session.framesDir, '%04d.jpg'),
    '-frames:v', String(count),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    '-progress', 'pipe:1',
    '-y', '-nostats',
    outPath,
  ]);

  ffmpeg.stdout.on('data', chunk => {
    const m = chunk.toString().match(/frame=\s*(\d+)/);
    if (m) send({ frame: parseInt(m[1]), total: count });
  });

  ffmpeg.on('close', code => {
    if (code !== 0) { send({ error: `ffmpeg encerrou com código ${code}` }); return res.end(); }
    send({ done: true, file: filename });
    res.end();
  });

  req.on('close', () => { try { ffmpeg.kill(); } catch {} });
});

// ── Crop download ─────────────────────────────────────────────────────────────

app.get('/api/crop/download/:file', (req, res) => {
  if (!session) return res.status(400).end();
  const filename = path.basename(req.params.file);
  const filePath = path.join(session.sessionDir, filename);
  if (!fs.existsSync(filePath)) return res.status(404).end();
  res.download(filePath);
});

// ── Start ────────────────────────────────────────────────────────────────────

function localIPs() {
  const ips = [];
  for (const ifaces of Object.values(os.networkInterfaces())) {
    for (const i of ifaces) {
      if (i.family === 'IPv4' && !i.internal) ips.push(i.address);
    }
  }
  return ips;
}

app.listen(PORT, '0.0.0.0', () => {
  console.log('\nTrackNet · Anotador');
  console.log(`  Local : http://localhost:${PORT}`);
  localIPs().forEach(ip => console.log(`  LAN   : http://${ip}:${PORT}`));
  console.log(`  Vídeos: ${VIDEOS_DIR}\n`);
});
