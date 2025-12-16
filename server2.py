# server.py — multi-bot Realtime voice chat (7 preset scenarios)
# --------------------------------------------------------------
# Run:
#   pip install flask flask-cors requests python-dotenv
#   python server.py
# Open: http://127.0.0.1:5000/realtime
#
# Edit the BOTS list below to customize role/task/constraints per button.

import os
import json
import textwrap
import requests
import tempfile
import statistics
import assemblyai as aai
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
from flask import Flask, request, jsonify, Response, redirect
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# --------------------------- Config ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
OPENAI_REALTIME_VOICE_DEFAULT = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
RT_SILENCE_MS = int(os.getenv("RT_SILENCE_MS", "1200"))  # pause after user stops
VAD_THRESHOLD = float(os.getenv("RT_VAD_THRESHOLD", "0.5"))
TURN_PREFIX_PADDING_MS = int(os.getenv("RT_PREFIX_PADDING_MS", "300"))

# 7 preset "bots". Edit freely.
BOTS = [
    {
        "id": "apt-en",
        "title": "Order Breakfast (EN)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
        "role": "Patient Polish conversation partner",
        "task": (
            "Task Situation: It’s Monday morning at 8 a.m. in October. You are on your way to school and decide to pick up breakfast for yourself and your friend. You have just entered a coffee shop. Your Goal: Order any food and a drink you want. Order food and a drink for your friend. Your friend loves bagels and lattes but cannot have dairy products (e.g., cow’s milk)."
        ),
        "constraints": (
            "Speak slowly; use novice-high vocabulary. Track learner errors; if an utterance has 3+ mistakes, "
            "signal partial misunderstanding and ask them to repeat or rephrase, then scaffold."
        ),
        "language_hint": "English"
    },
    {
        "id": "rest-es",
        "title": "Restaurant Booking (ES)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
        "role": "Compañero de conversación en español",
        "task": (
            "Practicar reservas y pedidos en un restaurante: saludar, número de personas, hora, alergias, "
            "preferencias y cuenta."
        ),
        "constraints": (
            "Habla claro y despacio; vocabulario de nivel intermedio-bajo. Negocia significado si hay 3+ errores; "
            "pide repetir con una pista."
        ),
        "language_hint": "Spanish"
    },
    {
        "id": "doctor-en",
        "title": "Doctor Visit (ZH)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
        "role": "Supportive clinic intake partner",
        "task": (
            "Simulate a primary-care intake: symptoms, duration, severity, medications, allergies, history. "
            "Encourage precise descriptions and safety-seeking behavior."
        ),
        "constraints": "Speak calmly; define medical terms briefly; check comprehension; novice-high register.",
        "language_hint": "Chinese"
    },
    {
        "id": "travel-es",
        "title": "Travel Booking (IT)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
      "role": "Agente di viaggi",
"task": (
    "Aiuta a prenotare un viaggio: date, destinazioni, budget, alloggio, trasporto. Rafforza numeri, "
    "date e conferme."
),
"constraints": "Ritmo lento; riformula quando c'è confusione; conferma i dati chiave.",
        "language_hint": "Italian"
    },
    {
        "id": "job-en",
        "title": "Job Interview (FR)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
       "role": "Coach d'entretien",
"task": (
    "Mène un entretien simulé : expérience, compétences, exemples STAR, relances. Offre un bref retour après chaque "
    "réponse et un résumé en trois points à la fin."
),
"constraints": "Utilise un vocabulaire accessible ; garde des tours de parole courts ; une question à la fois.",
        "language_hint": "French"
    },
    {
        "id": "roommate-en",
        "title": "Roommate Negotiation (ES)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
        "role": "Collaborative roommate",
        "task": (
            "Negotiate apartment norms: cleaning, guests, noise, shared costs. Seek agreement with proposals and "
            "counterproposals; confirm decisions."
        ),
        "constraints": "Slow pace, novice-high; negotiate meaning after 3+ mistakes.",
        "language_hint": "Spanish"
    },
    {
        "id": "returns-es",
        "title": "Customer Service Return (ES)",
        "voice": OPENAI_REALTIME_VOICE_DEFAULT,
        "role": "Agente de atención al cliente",
        "task": (
            "Practica devoluciones/cambios: saludar, describir problema, ticket, política, opciones. Modela cortesía y "
            "frases útiles."
        ),
        "constraints": "Habla despacio, frases cortas; comprueba comprensión; pide repetir tras 3+ errores.",
        "language_hint": "Spanish"
    },
]

BOT_MAP = {b["id"]: b for b in BOTS}

# ---------------------- Prompt builder ------------------------

def build_system_prompt(bot: dict) -> str:
    base = textwrap.dedent(f"""
    ROLE: {bot['role']}
    TASK: {bot['task']}
    CONSTRAINTS: {bot['constraints']}
    STYLE: Conversational, concise, interactive. Keep turns short; end most turns with a brief, relevant question. But also add personal information dependent on your role. 
    VOICE/LANGUAGE: Speak primarily in {bot['language_hint']}. If the learner switches language, mirror briefly then steer back.
    ERROR HANDLING: Track learner errors in each utterance; if 3+ issues (grammar/lexis/pronunciation leading to ambiguity), be strict!!! on pronunciation,
      politely signal misunderstanding and ask for a clear repeat or rephrase, offering a simple model. 
    RECAP: When the scenario goals are completed, give a 2–3 bullet summary and suggest one actionable next step.
    """).strip()
    return base

def transcribe_with_utterance_metrics(file_path: str) -> dict:
    if not aai.settings.api_key:
        return {"error": "No ASSEMBLYAI_API_KEY set"}

    config = aai.TranscriptionConfig(
        speaker_labels=True,   # diarization
        punctuate=True,
        format_text=True
    )
    transcriber = aai.Transcriber()
    t = transcriber.transcribe(file_path, config=config)
    if t.status != aai.TranscriptStatus.completed:
        raise RuntimeError(f"Transcription failed: {t.error}")

    utterances_out = []
    for utt in (t.utterances or []):
        word_confs = [w.confidence for w in (utt.words or []) if w.confidence is not None]
        avg_conf = round(statistics.mean(word_confs), 3) if word_confs else None
        utterances_out.append({
            "speaker": utt.speaker,          # "A", "B", ...
            "start_ms": utt.start,
            "end_ms": utt.end,
            "duration_ms": (utt.end - utt.start) if utt.end and utt.start else None,
            "text": utt.text,
            "avg_confidence": avg_conf,
            "words": [
                {
                    "text": w.text,
                    "start_ms": w.start,
                    "end_ms": w.end,
                    "confidence": round(w.confidence, 3) if w.confidence is not None else None
                } for w in (utt.words or [])
            ]
        })

    by_speaker = {}
    for r in utterances_out:
        s = r["speaker"]
        by_speaker.setdefault(s, {"utterances": 0, "accum": []})
        by_speaker[s]["utterances"] += 1
        if r["avg_confidence"] is not None:
            by_speaker[s]["accum"].append(r["avg_confidence"])
    for s, d in by_speaker.items():
        d["mean_confidence"] = round(statistics.mean(d["accum"]), 3) if d["accum"] else None
        d.pop("accum", None)

    return {
        "audio_duration_ms": int(t.audio_duration * 1000) if t.audio_duration else None,
        "utterances": utterances_out,
        "by_speaker": by_speaker
    }

# ------------------------ Flask app ---------------------------
app = Flask(__name__)
CORS(app)


@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    suffix = os.path.splitext(f.filename or ".webm")[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        path = tmp.name
    try:
        data = transcribe_with_utterance_metrics(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(path)
        except: pass


@app.route("/")
def index():
    return redirect("/realtime", code=302)

@app.route("/session", methods=["POST"])
def session():
    if not OPENAI_API_KEY:
        return jsonify({"error": "Missing OPENAI_API_KEY"}), 400

    body = request.get_json(silent=True) or {}
    bot_id = body.get("bot_id") or request.args.get("bot") or BOTS[0]["id"]
    bot = BOT_MAP.get(bot_id, BOTS[0])

    system_prompt = build_system_prompt(bot)

    # Mint an ephemeral client_secret for WebRTC (valid ~1 minute)
    try:
        r = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1",
            },
            json={
                "model": OPENAI_REALTIME_MODEL,
                "voice": bot.get("voice", OPENAI_REALTIME_VOICE_DEFAULT),
                "instructions": system_prompt,
                "modalities": ["audio", "text"],
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": VAD_THRESHOLD,
                    "silence_duration_ms": RT_SILENCE_MS,
                    "prefix_padding_ms": TURN_PREFIX_PADDING_MS,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        # Augment response with our chosen model/voice/bot
        data.update({
            "model": OPENAI_REALTIME_MODEL,
            "bot": {k: bot[k] for k in ("id", "title")},
        })
        return jsonify(data)
    except requests.HTTPError as e:
        return jsonify({"error": f"OpenAI error {e.response.status_code}", "details": e.response.text}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/realtime")
def realtime():
    # Inject a public view of the bots (id + title only)
    public = [{"id": b["id"], "title": b["title"]} for b in BOTS]
    html = (
        REALTIME_HTML
        .replace("__BOTS_JSON__", json.dumps(public, ensure_ascii=False))
        .replace("__VAD_THRESHOLD__", json.dumps(VAD_THRESHOLD))
        .replace("__RT_SILENCE_MS__", str(RT_SILENCE_MS))
        .replace("__RT_PREFIX_PADDING_MS__", str(TURN_PREFIX_PADDING_MS))
    )
    return Response(html, mimetype="text/html")

# -------------------------- HTML -----------------------------
REALTIME_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Voice Chat (Realtime API)</title>
  <style>
    html,body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b1020;color:#e9ecf1;margin:0}
    .wrap{max-width:880px;margin:0 auto;padding:24px}
    .card{background:#141a2f;border:1px solid #1f2745;border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    button{background:#35b26f;color:#fff;border:none;border-radius:999px;padding:10px 16px;font-weight:600;cursor:pointer}
    button.stop{background:#e9534a}
    button.ghost{background:transparent;border:1px solid #2b335a}
    .status{margin-left:auto;font-size:.85rem;opacity:.85}
    .log{background:#0e1428;border:1px solid #1f2745;border-radius:12px;padding:12px;height:320px;overflow:auto;font-size:.95rem;margin-top:12px}
    .msg{margin:6px 0}.user{color:#8fd3ff}.assistant{color:#b0ffa3}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0 12px}
    .chip{background:#0e1428;border:1px solid #2b335a;border-radius:999px;padding:8px 12px;cursor:pointer;text-align:center}
    .chip.active{background:#224a39;border-color:#35b26f}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Realtime Voice (WebRTC)</h1>
    <p>Select a scenario, then connect. The agent will speak and stream text. Switch scenarios any time.</p>

    <div class="card">
      <div class="row">
        <button id="connect">Connect</button>
        <button id="disconnect" class="stop" disabled>Disconnect</button>
        <button id="nudge" class="ghost" disabled>Push-to-talk</button>
        <button id="clear" class="ghost">Clear log</button>
        <button id="next" class="ghost">Next scenario</button>
        <button id="recStart" class="ghost" disabled>Start Rec</button>
		<button id="recStop" class="ghost" disabled>Stop Rec</button>
		<button id="recAnalyze" class="ghost" disabled>Analyze</button>
        <span id="status" class="status">idle</span>
      </div>
      <div id="scenarioBar" class="grid"></div>
      <div class="row" style="gap:8px;margin:4px 0 8px">
        <span>Selected:</span> <b id="selectedTitle">(none)</b>
      </div>
      <audio id="remote" autoplay playsinline></audio>
      <div id="log" class="log" aria-live="polite"></div>
    </div>
  </div>

<script id="bots" type="application/json">__BOTS_JSON__</script>
<script>
const bots = JSON.parse(document.getElementById('bots').textContent);
const logEl = document.getElementById('log');
const connectBtn = document.getElementById('connect');
const disconnectBtn = document.getElementById('disconnect');
const nudgeBtn = document.getElementById('nudge');
const clearBtn = document.getElementById('clear');
const nextBtn = document.getElementById('next');
const scenarioBar = document.getElementById('scenarioBar');
const selectedTitleEl = document.getElementById('selectedTitle');
const remoteAudio = document.getElementById('remote');
const statusEl = document.getElementById('status');
const recStartBtn = document.getElementById('recStart');
const recStopBtn = document.getElementById('recStop');
const recAnalyzeBtn = document.getElementById('recAnalyze');
const TURN_DETECTION_CONFIG = {
  type: 'server_vad',
  threshold: __VAD_THRESHOLD__,
  silence_duration_ms: __RT_SILENCE_MS__,
  prefix_padding_ms: __RT_PREFIX_PADDING_MS__,
  create_response: true,
  interrupt_response: true,
};
const INPUT_TRANSCRIPTION_MODEL = "whisper-1";

recStartBtn.onclick = async ()=>{ await startRecorder(); recStartBtn.disabled=true; recStopBtn.disabled=false; recAnalyzeBtn.disabled=true; };
recStopBtn.onclick = async ()=>{ await stopRecorder(); recStartBtn.disabled=false; recStopBtn.disabled=true; recAnalyzeBtn.disabled=false; };
recAnalyzeBtn.onclick = async ()=>{ await analyzeRecording(); recAnalyzeBtn.disabled=true; };

function enableRecButtons(ready){
  recStartBtn.disabled = !ready;
  recStopBtn.disabled = true;
  recAnalyzeBtn.disabled = true;
}

// After successful connect:
async function connect(){
  // ... existing code ...
  // at the end of a successful connect:
  enableRecButtons(true);
}
async function disconnect(){
  // ... existing code ...
  enableRecButtons(false);
}


let pc, dc, micStream;
let selectedBotId = bots[0]?.id || null;

// Show assistant only after it finishes speaking
const SHOW_AGENT_AFTER_SPEAKS = true;

// ---- Gate: track whether a user turn is still open (transcription not DONE yet)
let userTurnOpen = false;
function isUserAudioItem(it){
  return it && it.role === 'user' && it.type === 'message' &&
         Array.isArray(it.content) && it.content.some(p => p && p.type === 'input_audio');
}

// ---- Buffer + gate assistant output until user transcript is done (or timeout)
function createAgentBuffer(forward) {
  let haveAudio = false, audio = '', text = '', flushed = false;
  const MAX_WAIT_MS = 1800;     // how long to wait for user transcript to finish
  const STEP_MS     = 100;

  function flushWhenReady() {
    if (flushed) return;
    const start = Date.now();
    (function attempt(){
      if (!userTurnOpen || (Date.now() - start) >= MAX_WAIT_MS) {
        const line = (audio || text || '').trim();
        if (line) forward({ type: '__agent.buffer.flush', text: line });
        flushed = true;
        return;
      }
      setTimeout(attempt, STEP_MS);
    })();
  }

  return (msg) => {
    if (!SHOW_AGENT_AFTER_SPEAKS) return forward(msg);

    // Track user-turn lifecycle for gating
    // Also gate on raw VAD events to open/close faster
    if (msg.type === 'input_audio_buffer.speech_started') { userTurnOpen = true; return forward(msg); }
    if (msg.type === 'input_audio_buffer.committed') { userTurnOpen = false; return forward(msg); }

    if (msg.type === 'conversation.item.created') {
      const it = msg.item || {};
      if (isUserAudioItem(it)) userTurnOpen = true;
      return forward(msg);
    }
    if (msg.type === 'conversation.item.input_audio_transcription.completed' ||
        msg.type === 'conversation.item.input_audio_transcription.done') {
      userTurnOpen = false;
      return forward(msg);
    }

    // Assistant buffering
    switch (msg.type) {
      case 'response.created':
        haveAudio = false; audio = ''; text = ''; flushed = false;
        return forward(msg);

      case 'response.audio_transcript.delta':
        haveAudio = true; audio += (msg.delta || '');
        return; // swallow while buffering

      case 'response.audio_transcript.done':
        return flushWhenReady(); // print after your turn is closed (or timeout)

      case 'response.output_text.delta':
        text += (msg.delta || '');
        return; // swallow while buffering

      case 'response.output_text.done':
        if (!haveAudio) return flushWhenReady();
        return;

      case 'response.done':
        // safety: ensure we flush even if we missed the earlier signals
        flushWhenReady();
        return forward(msg);

      default:
        return forward(msg);
    }
  };
}

// --- helpers ---
function append(who, text){
  const d=document.createElement('div');
  d.className = 'msg ' + who;
  d.textContent = (who==='assistant'?'Agent':'You') + ': ' + text;
  logEl.appendChild(d); logEl.scrollTop = logEl.scrollHeight;
}
function setStatus(t){ statusEl.textContent = t; }
function selectBot(botId){
  selectedBotId = botId;
  for (const btn of scenarioBar.querySelectorAll('.chip')) btn.classList.toggle('active', btn.dataset.id===botId);
  const b = bots.find(x=>x.id===botId); selectedTitleEl.textContent = b? b.title : '(none)';
}
function buildScenarioButtons(){
  scenarioBar.innerHTML = '';
  bots.forEach((b,i)=>{
    const el = document.createElement('div');
    el.className = 'chip' + (i===0?' active':'');
    el.dataset.id = b.id;
    el.textContent = b.title;
    el.onclick = ()=> selectBot(b.id);
    scenarioBar.appendChild(el);
  });
  selectBot(selectedBotId);
}

function waitForIceGatheringComplete(pc) {
  return new Promise(resolve => {
    if (pc.iceGatheringState === 'complete') return resolve();
    function check() {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    }
    pc.addEventListener('icegatheringstatechange', check);
  });
}

// --- streaming render state ---
let asstEl=null, asstBuf='', asstMode='audio'; // 'audio' or 'text'
let userEl=null, userBuf='';



function handleOAIEvent(msg) {
  switch (msg.type) {
    // Assistant OUTPUT (audio transcript stream)
    case 'response.audio_transcript.delta': {
      if (!asstEl || asstMode!=='audio') { asstEl=document.createElement('div'); asstEl.className='msg assistant'; logEl.appendChild(asstEl); asstMode='audio'; asstBuf=''; }
      asstBuf += (msg.delta||'');
      asstEl.textContent = 'Agent: ' + asstBuf; logEl.scrollTop = logEl.scrollHeight; break;
    }
    case 'response.audio_transcript.done': {
      if (asstEl) asstEl.textContent = 'Agent: ' + (msg.transcript||asstBuf);
      asstEl=null; asstBuf=''; asstMode='audio'; setStatus('ready'); break;
    }

    // Assistant OUTPUT (plain text stream)
    case 'response.output_text.delta': {
      if (!asstEl || asstMode!=='text') { asstEl=document.createElement('div'); asstEl.className='msg assistant'; logEl.appendChild(asstEl); asstMode='text'; asstBuf=''; }
      asstBuf += (msg.delta||'');
      asstEl.textContent = 'Agent: ' + asstBuf; logEl.scrollTop = logEl.scrollHeight; break;
    }
    case 'response.output_text.done': {
      if (asstEl) asstEl.textContent = 'Agent: ' + (msg.text||asstBuf);
      asstEl=null; asstBuf=''; asstMode='audio'; setStatus('ready'); break;
    }

    // Your INPUT (mic) transcription
    case 'conversation.item.input_audio_transcription.delta': {
      if (!userEl) { userEl=document.createElement('div'); userEl.className='msg user'; logEl.appendChild(userEl); userBuf=''; }
      userBuf += (msg.delta||'');
      userEl.textContent = 'You: ' + userBuf; logEl.scrollTop = logEl.scrollHeight; break;
    }
    case 'conversation.item.input_audio_transcription.completed':
    case 'conversation.item.input_audio_transcription.done': {
      if (userEl) userEl.textContent = 'You: ' + (msg.transcript||userBuf);
      userEl=null; userBuf=''; break;
    }
    case '__agent.buffer.flush': {
  append('assistant', (msg.text || '').trim());
  break;
}

case 'conversation.item.created': {
  // If this is the start of a user turn with input_audio,
  // create a placeholder "You:" line immediately so it precedes the agent.
  const it = msg.item || {};
  if (it.role === 'user' && it.type === 'message' && Array.isArray(it.content)) {
    const hasAudio = it.content.some(p => p && p.type === 'input_audio');
    if (hasAudio && !userEl) {
      userEl = document.createElement('div');
      userEl.className = 'msg user';
      userEl.textContent = 'You: ';   // blank placeholder (no mic emoji)
      logEl.appendChild(userEl);
      logEl.scrollTop = logEl.scrollHeight;
      userBuf = '';
    }
  }
  break;
}

    // Status only
    case 'input_audio_buffer.speech_started': {
  userTurnOpen = true;
  if (!userEl) {
    userEl = document.createElement('div');
    userEl.className = 'msg user';
    userEl.textContent = 'You: ';
    logEl.appendChild(userEl);
    logEl.scrollTop = logEl.scrollHeight;
    userBuf = '';
  }
  setStatus('listening…');
  break;
}
    
    case 'input_audio_buffer.committed': {
      userTurnOpen = false;
      break;
    }
case 'input_audio_buffer.speech_stopped': setStatus('processing…'); break;

    case 'session.created': setStatus('connected'); break;
    default: /* ignore */ break;
  }
}

function wireDataChannel(dc) {
  dc.onopen = () => {
    dc.send(JSON.stringify({
      type: 'session.update',
      session: {
        modalities: ['audio','text'],
        input_audio_transcription: { model: INPUT_TRANSCRIPTION_MODEL },
        turn_detection: TURN_DETECTION_CONFIG
      }
    }));
    setStatus('connected');
  };

  const deliver = (m) => handleOAIEvent(m);
  const bufferedDeliver = createAgentBuffer(deliver);

  dc.onmessage = (e) => {
    try { bufferedDeliver(JSON.parse(e.data)); } catch { /* ignore non-JSON */ }
  };
}

async function connect(){
  connectBtn.disabled = true;
  try{
    // 1) Ask server for ephemeral token with selected bot
    const sessRes = await fetch('/session', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ bot_id: selectedBotId }) });
    const session = await sessRes.json();
    if (!sessRes.ok) throw new Error(session.error || 'Session error');

    // 2) Mic
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // 3) WebRTC peer connection
    pc = new RTCPeerConnection();
    pc.addTransceiver('audio', { direction: 'recvonly' }); // receive audio
    pc.ontrack = (e) => { remoteAudio.srcObject = e.streams[0]; };
    for (const track of micStream.getTracks()) pc.addTrack(track, micStream); // send mic

    // 4) Data channel for commands/events
    dc = pc.createDataChannel('oai-events');
    wireDataChannel(dc);

    // 5) Offer
    const offer = await pc.createOffer({ offerToReceiveAudio: true });
    await pc.setLocalDescription(offer);
    await waitForIceGatheringComplete(pc);

    // 6) Handshake with Realtime
    const url = `https://api.openai.com/v1/realtime?model=${encodeURIComponent(session.model || 'gpt-4o-realtime-preview-2024-12-17')}`;
    const ans = await fetch(url, {
      method: 'POST',
      body: pc.localDescription.sdp,
      headers: {
        'Authorization': `Bearer ${session.client_secret?.value || session.client_secret || ''}`,
        'Content-Type': 'application/sdp',
        'OpenAI-Beta': 'realtime=v1'
      }
    });
    const sdpText = await ans.text();
    if (!ans.ok) { append('assistant', 'Realtime handshake failed: ' + sdpText); throw new Error('Realtime SDP error'); }
    const answer = { type: 'answer', sdp: sdpText };
    await pc.setRemoteDescription(answer);

    connectBtn.disabled = true;
    disconnectBtn.disabled = false;
    nudgeBtn.disabled = false;
    setStatus('ready');
    append('assistant', 'Connected. Speak when you are ready');
  }catch(e){
    connectBtn.disabled = false;
    setStatus('error');
    append('assistant', 'Connect error: ' + e.message);
    console.error(e);
  }
}

async function disconnect(){
  nudgeBtn.disabled = true; disconnectBtn.disabled = true; connectBtn.disabled = false;
  if (dc) try{ dc.close(); }catch(e){}
  if (pc) try{ pc.close(); }catch(e){}
  if (micStream) for (const t of micStream.getTracks()) t.stop();
  setStatus('idle');
}

// Manual poke (if VAD is shy)
nudgeBtn.addEventListener('click', ()=>{
  if (!dc || dc.readyState !== 'open') return;
  dc.send(JSON.stringify({ type: 'response.create', response: { modalities: ['audio','text'] } }));
  append('user', '⏺️ Nudge sent (audio+text requested).');
});

clearBtn.addEventListener('click', ()=>{ logEl.innerHTML=''; });
nextBtn.addEventListener('click', ()=>{
  const idx = bots.findIndex(b=>b.id===selectedBotId);
  const next = bots[(idx+1) % bots.length];
  selectBot(next.id);
});

connectBtn.addEventListener('click', connect);
disconnectBtn.addEventListener('click', disconnect);

buildScenarioButtons();

// --- Recording (mix mic + remote audio) ---
let recorder, mixedStream, mixCtx, mixDest;
let recordedChunks = [];

async function startRecorder() {
  // Need both: micStream (already created) and remoteAudio.srcObject (after connect)
  if (!micStream || !remoteAudio.srcObject) {
    append('assistant', 'Recorder not ready (need mic + remote). Connect first and wait for audio.');
    return;
  }
  if (recorder && recorder.state === 'recording') return;

  mixCtx = new (window.AudioContext || window.webkitAudioContext)();
  const micNode = mixCtx.createMediaStreamSource(micStream);
  const remoteNode = mixCtx.createMediaStreamSource(remoteAudio.srcObject);
  mixDest = mixCtx.createMediaStreamDestination();
  micNode.connect(mixDest);
  remoteNode.connect(mixDest);

  mixedStream = mixDest.stream;
  recordedChunks = [];
  recorder = new MediaRecorder(mixedStream, { mimeType: 'audio/webm' });
  recorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  recorder.start();
  append('assistant', 'Recording started (mic + agent).');
}

async function stopRecorder() {
  if (!recorder || recorder.state !== 'recording') return null;
  await new Promise(res => {
    recorder.onstop = res;
    recorder.stop();
  });
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  append('assistant', `Recording stopped (${Math.round(blob.size/1024)} KB).`);
  try { mixCtx && mixCtx.close(); } catch {}
  return blob;
}

async function analyzeRecording() {
  const blob = await stopRecorder();
  if (!blob) { append('assistant', 'Nothing to analyze.'); return; }
  const fd = new FormData();
  fd.append('file', blob, 'session.webm');
  const r = await fetch('/analyze_audio', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { append('assistant', 'Analyze error: ' + (j.error || r.status)); return; }
  renderUtteranceTable(j);
}

// --- Simple utterance table renderer ---
function renderUtteranceTable(report){
  const { utterances = [], by_speaker = {}, audio_duration_ms } = report || {};
  let html = `<div class="msg assistant">Analysis: duration ${audio_duration_ms ?? '—'} ms</div>`;
  html += `<table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:.9rem">
    <thead><tr>
      <th style="text-align:left;border-bottom:1px solid #2b335a;padding:4px">#</th>
      <th style="text-align:left;border-bottom:1px solid #2b335a;padding:4px">Spk</th>
      <th style="text-align:left;border-bottom:1px solid #2b335a;padding:4px">Start</th>
      <th style="text-align:left;border-bottom:1px solid #2b335a;padding:4px">End</th>
      <th style="text-align:left;border-bottom:1px solid #2b335a;padding:4px">Avg Conf</th>
      <th style="text-align:left;border-bottom:1px solid #2b335a;padding:4px">Text</th>
    </tr></thead><tbody>`;
  utterances.forEach((u,i)=>{
    html += `<tr>
      <td style="padding:4px;border-bottom:1px solid #2b335a">${i+1}</td>
      <td style="padding:4px;border-bottom:1px solid #2b335a">${u.speaker ?? ''}</td>
      <td style="padding:4px;border-bottom:1px solid #2b335a">${u.start_ms ?? ''}</td>
      <td style="padding:4px;border-bottom:1px solid #2b335a">${u.end_ms ?? ''}</td>
      <td style="padding:4px;border-bottom:1px solid #2b335a">${u.avg_confidence ?? ''}</td>
      <td style="padding:4px;border-bottom:1px solid #2b335a">${(u.text || '').replace(/</g,'&lt;')}</td>
    </tr>`;
  });
  html += `</tbody></table>`;

  // speaker summary
  if (by_speaker && Object.keys(by_speaker).length){
    html += `<div style="margin-top:6px" class="msg assistant">By speaker: ${
      Object.entries(by_speaker).map(([s,v])=>`${s}: ${v.utterances} utts, mean conf ${v.mean_confidence ?? '—'}`).join(' • ')
    }</div>`;
  }

  const wrap = document.createElement('div');
  wrap.innerHTML = html;
  logEl.appendChild(wrap);
  logEl.scrollTop = logEl.scrollHeight;
}

</script>
</body>
</html>
"""

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)
