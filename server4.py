import os
import json
import sys
from flask import Flask, request, Response, redirect

# Import complexity analysis functions
try:
    from analyze_complexity import analyze_complexity
except ImportError:
    # If import fails, we'll handle it in the analyze endpoint
    analyze_complexity = None

# ---------- Config ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set; /session will fail.")

MODEL = "gpt-realtime-mini-2025-12-15"
SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
REALTIME_URL = "https://api.openai.com/v1/realtime"

# Seven preset bots / scenarios
BOTS = [
    {
        "id": "apt_en",
        "label": "Renting an apartment (EN)",
        "role": "Patient English conversation partner",
        "task": (
            "The learner will ask questions about renting an apartment. "
            "Help them think about cost, location, roommates, number of rooms, "
            "and number of bathrooms, and guide them to a decision. Don't be afraid to add your own opinions about things, and try not to ask a question everytime."
        ),
        "constraints": (
            "Speak clearly and a bit slowly. Use vocabulary appropriate for an upper-intermediate learner. "
            "Please don't guess the context, if you hear one word that is wrong, please tell the learner and then continue with the task"
            "When the learner makes incomprehensible language, or says something you are not expecting, signal that you did not "
            "fully understand and ask them to repeat or clarify. Be very strict with this, the language should only be in English. You don't understand anything else. It's very important that you signal that you don't understand if the language is unclear, this helps learning and is what a good tutor does. Be a good tutor."
        ),
    },
    {
        "id": "apt_es",
        "label": "Alquilar un piso (ES)",
        "role": "Compañero de conversación en español",
        "task": (
            "Habla con el estudiante sobre cómo alquilar un piso: presupuesto, barrio, tamaño, "
            "compañeros de piso y transporte."
        ),
        "constraints": (
            "Usa español claro, ritmo lento y vocabulario de nivel intermedio. "
            "Reformula suavemente los errores graves y haz preguntas de seguimiento."
        ),
    },
    {
        "id": "travel",
        "label": "Planning a trip",
        "role": "Friendly travel planner",
        "task": "Help the learner plan a short trip, asking about budget, interests, accommodation, and transport.",
        "constraints": "Keep answers short and interactive, always ending with a question that moves the planning forward.",
    },
    {
        "id": "job",
        "label": "Job interview practice",
        "role": "Supportive interviewer",
        "task": "Simulate a job interview in the learner's target language. Ask typical interview questions and give brief feedback.",
        "constraints": "Stay encouraging. If answers are unclear, ask the learner to rephrase rather than correcting directly.",
    },
    {
        "id": "restaurant",
        "label": "Restaurant role-play",
        "role": "Restaurant server",
        "task": (
            "Role-play ordering food in a restaurant. Ask follow-up questions about allergies, preferences, and payment."
        ),
        "constraints": "Use simple language and lots of repetition. Keep the conversation practical and concrete.",
    },
    {
        "id": "debate",
        "label": "Light debate",
        "role": "Discussion partner",
        "task": "Hold a light, friendly debate on a topic the learner chooses. Encourage them to justify opinions.",
        "constraints": (
            "Avoid sensitive political topics. Focus on language development, asking for clarification and examples."
        ),
    },
    {
        "id": "free_talk",
        "label": "Free conversation",
        "role": "Curious conversation partner",
        "task": "Have an open-ended conversation driven by the learner's interests.",
        "constraints": "Keep turns short, ask many follow-up questions, and adapt your language level to the learner.",
    },
]

# ---------- Flask app ----------

app = Flask(__name__)


@app.route("/")
def root():
    # Convenience: go straight to UI
    return redirect("/realtime")


@app.route("/favicon.ico")
@app.route("/apple-touch-icon.png")
@app.route("/apple-touch-icon-precomposed.png")
def icons():
    """
    Quietly satisfy browsers' automatic icon requests so logs don't show 404s.
    """
    return Response(status=204)


@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    """
    Chrome DevTools pings this path automatically; return 204 so it doesn't clutter logs.
    """
    return Response(status=204)


@app.route("/realtime")
def realtime_page():
    html = REALTIME_HTML.replace("{MODEL}", MODEL).replace(
        "{BOTS_JSON}", json.dumps(BOTS)
    )
    return Response(html, mimetype="text/html")


@app.route("/session", methods=["POST"])
def create_session():
    """
    Server endpoint: creates a Realtime session with instructions for the chosen bot
    and returns the full session JSON to the browser (including ephemeral key).
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        bot_id = data.get("bot_id")
        bot = next((b for b in BOTS if b["id"] == bot_id), BOTS[0])
    except Exception:
        bot = BOTS[0]

    instructions = (
        f"ROLE: {bot['role']}\n"
        f"TASK: {bot['task']}\n"
        f"CONSTRAINTS: {bot['constraints']}\n"
        "STYLE: Conversational, concise, interactive. End most turns with a short, relevant question.\n"
        "VOICE/LANGUAGE: Speak in the learner's language. Prefer their target language when they use it."
    )

    import requests

    try:
        resp = requests.post(
            SESSION_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1",
            },
        json={
            "model": MODEL,
            "voice": "alloy",
            "instructions": instructions,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            # 👇 NEW: tell the session to transcribe user audio with Whisper
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 300,
                "create_response": True,
                "interrupt_response": True,
            },
        },
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to create session: {e}"}, 500

    return Response(resp.text, mimetype="application/json")


@app.route("/analyze", methods=["POST"])
def analyze_transcript():
    """
    Analyzes a transcript for accuracy and language complexity.
    Uses Python script for complexity metrics, ChatGPT for accuracy analysis.
    """
    import requests
    
    try:
        data = request.get_json(force=True, silent=True) or {}
        transcript = data.get("transcript", "")
        
        if not transcript:
            return {"error": "No transcript provided"}, 400
        
        # Calculate complexity metrics using Python script
        complexity_results = {}
        if analyze_complexity:
            try:
                # Run complexity analysis (quiet mode to avoid duplicate output)
                complexity_results = analyze_complexity(transcript, verbose=False)
                # Debug: print what we got
                print(f"DEBUG: Complexity results: {complexity_results}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Complexity analysis failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                complexity_results = {}
        else:
            print("Warning: analyze_complexity module not available", file=sys.stderr)
        
        # Use ChatGPT for accuracy analysis (grammar/vocabulary errors)
        if not OPENAI_API_KEY:
            # If no API key, return only complexity results
            return {
                "analysis": {
                    "accuracy": {
                        "score": None,
                        "grammar_errors": [],
                        "vocabulary_errors": [],
                        "overall_assessment": "Accuracy analysis requires OPENAI_API_KEY"
                    },
                    "complexity": {
                        "ttr": complexity_results.get("ttr"),
                        "lexical_density": complexity_results.get("lexical_density"),
                        "mean_tunit_length": complexity_results.get("mean_tunit_length"),
                        "clauses_per_tunit": complexity_results.get("clauses_per_tunit")
                    }
                }
            }, 200
        
        # Prepare analysis prompt for accuracy only
        analysis_prompt = f"""Analyze the following language learning conversation transcript. Focus on the USER's utterances (lines starting with "You:"). Provide a detailed analysis in JSON format with the following structure:

{{
  "accuracy": {{
    "score": <0-100>,
    "grammar_errors": ["error 1: description with correction", "error 2: description with correction"],
    "vocabulary_errors": ["error 1: description with correction", "error 2: description with correction"],
    "overall_assessment": "<brief assessment>"
  }}
}}

IMPORTANT:
- grammar_errors and vocabulary_errors must be arrays of strings, even if empty. Each string should describe the error and provide a correction.
- Only analyze the user's speech, ignore the agent's responses.

Transcript:
{transcript}

Provide only valid JSON, no additional text or markdown formatting."""

        # Call OpenAI API for accuracy analysis only
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert language learning tutor. Analyze transcripts and provide structured feedback in JSON format only. Return valid JSON without markdown code blocks or additional text. Ensure grammar_errors and vocabulary_errors are always arrays, even if empty."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        
        result = resp.json()
        analysis_text = result["choices"][0]["message"]["content"]
        
        # Clean up the response - remove markdown code blocks if present
        cleaned_text = analysis_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]  # Remove ```json
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]  # Remove ```
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]  # Remove closing ```
        cleaned_text = cleaned_text.strip()
        
        # Parse accuracy results
        try:
            accuracy_json = json.loads(cleaned_text)
            accuracy_data = accuracy_json.get("accuracy", {})
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return error
            accuracy_data = {
                "score": None,
                "grammar_errors": [],
                "vocabulary_errors": [],
                "overall_assessment": f"Error parsing accuracy analysis: {str(e)}"
            }
        
        # Combine accuracy (from ChatGPT) and complexity (from Python script)
        # Extract complexity values - handle both direct keys and nested structure
        ttr = complexity_results.get("ttr") if complexity_results else None
        lexical_density = complexity_results.get("lexical_density") if complexity_results else None
        mean_tunit_length = complexity_results.get("mean_tunit_length") if complexity_results else None
        clauses_per_tunit = complexity_results.get("clauses_per_tunit") if complexity_results else None
        
        # Debug output
        print(f"DEBUG: Extracted complexity - TTR: {ttr}, LD: {lexical_density}, MTL: {mean_tunit_length}, CPT: {clauses_per_tunit}", file=sys.stderr)
        
        combined_analysis = {
            "accuracy": accuracy_data,
            "complexity": {
                "ttr": ttr,
                "lexical_density": lexical_density,
                "mean_tunit_length": mean_tunit_length,
                "clauses_per_tunit": clauses_per_tunit
            }
        }
        
        print(f"DEBUG: Combined analysis complexity section: {combined_analysis['complexity']}", file=sys.stderr)
        
        return {"analysis": combined_analysis}, 200
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}, 500
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}, 500


# ---------- Static HTML / JS UI ----------

REALTIME_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Realtime Voice (WebRTC)</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0b1120;
      color: #e5e7eb;
      display: flex;
      align-items: stretch;
      justify-content: center;
      min-height: 100vh;
    }
    .app {
      margin: 24px;
      padding: 24px;
      max-width: 900px;
      width: 100%;
      background: radial-gradient(circle at top, #1d283a, #020617);
      border-radius: 18px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.7);
      border: 1px solid #1f2937;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 1.5rem;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    h1 span.badge {
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid #4b5563;
      color: #9ca3af;
    }
    p.subtitle {
      margin: 0;
      color: #9ca3af;
      font-size: 0.9rem;
    }
    .row {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    select {
      background: #020617;
      color: #e5e7eb;
      border-radius: 999px;
      border: 1px solid #374151;
      padding: 6px 10px;
      font-size: 0.85rem;
      outline: none;
    }
    button {
      border-radius: 999px;
      border: none;
      padding: 7px 14px;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #16a34a;
      color: #ecfdf5;
      box-shadow: 0 8px 16px rgba(22, 163, 74, 0.4);
      transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.08s ease, opacity 0.08s ease;
    }
    button:disabled {
      opacity: 0.5;
      cursor: default;
      box-shadow: none;
      transform: none;
    }
    button.ghost {
      background: transparent;
      color: #9ca3af;
      border: 1px solid #374151;
      box-shadow: none;
    }
    button.stop {
      background: #b91c1c;
      color: #fee2e2;
      box-shadow: 0 8px 16px rgba(185, 28, 28, 0.4);
    }
    button.secondary {
      background: #4b5563;
      color: #e5e7eb;
      box-shadow: 0 8px 16px rgba(75, 85, 99, 0.3);
    }
    button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    button:active:not(:disabled) {
      transform: translateY(0);
      box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    .status {
      font-size: 0.8rem;
      color: #9ca3af;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #4b5563;
      display: inline-block;
      margin-right: 6px;
    }
    .status-dot.live {
      background: #22c55e;
      box-shadow: 0 0 0 6px rgba(34, 197, 94, 0.25);
    }
    .log-wrap {
      margin-top: 8px;
      border-radius: 16px;
      background: rgba(15,23,42,0.9);
      border: 1px solid #111827;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      max-height: 420px;
    }
    .log-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 10px;
      font-size: 0.8rem;
      background: radial-gradient(circle at top left, #1f2937, #020617);
      border-bottom: 1px solid #111827;
    }
    .log-header .title {
      color: #9ca3af;
    }
    .log-header .actions {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .toggle {
      cursor: pointer;
      color: #9ca3af;
      font-size: 0.8rem;
      text-decoration: underline;
    }
    .log {
      padding: 10px;
      overflow-y: auto;
      font-size: 0.85rem;
    }
    .msg {
      margin-bottom: 6px;
      line-height: 1.4;
    }
    .msg.agent {
      color: #e5e7eb;
    }
    .msg.user {
      color: #a5b4fc;
    }
    .msg .label {
      font-weight: 600;
      margin-right: 4px;
    }
    .msg .meta {
      color: #6b7280;
      font-size: 0.75rem;
      margin-left: 4px;
    }
    .log.hidden {
      display: none;
    }
    audio {
      display: none;
    }
    .analysis-wrap {
      margin-top: 8px;
      border-radius: 16px;
      background: rgba(15,23,42,0.9);
      border: 1px solid #111827;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      max-height: 500px;
    }
    .analysis-content {
      padding: 16px;
      overflow-y: auto;
      font-size: 0.85rem;
    }
    .analysis-section {
      margin-bottom: 16px;
    }
    .analysis-section h3 {
      margin: 0 0 8px 0;
      font-size: 0.95rem;
      color: #a5b4fc;
    }
    .analysis-section p, .analysis-section ul {
      margin: 4px 0;
      color: #e5e7eb;
    }
    .analysis-section ul {
      padding-left: 20px;
    }
    .analysis-section li {
      margin: 4px 0;
    }
    .score {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      background: #16a34a;
      color: #ecfdf5;
      font-weight: 600;
      margin-left: 8px;
    }
    .level-badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      background: #4b5563;
      color: #e5e7eb;
      font-weight: 600;
      margin-left: 8px;
    }
    .analysis-loading {
      color: #9ca3af;
      font-style: italic;
    }
    .analysis-error {
      color: #ef4444;
    }
  </style>
</head>
<body>
  <div class="app">
    <div>
      <h1>Realtime Voice <span class="badge">WebRTC</span></h1>
      <p class="subtitle">
        Click <strong>Connect</strong>, speak, and the agent will reply with low-latency audio.
      </p>
    </div>

    <div class="row">
      <label for="scenario">Conversation:</label>
      <select id="scenario"></select>
    </div>

    <div class="row">
      <button id="connect">Connect</button>
      <button id="disconnect" class="stop" disabled>Disconnect</button>
      <button id="nudge" class="ghost" disabled>Tap to talk</button>
      <button id="clear" class="ghost">Clear transcript</button>
      <button id="analyze" class="secondary">Analyze transcript</button>
      <button id="download" class="ghost">Download transcript</button>
      <span class="status">
        <span id="statusDot" class="status-dot"></span>
        <span id="statusText">idle</span>
      </span>
    </div>

    <div class="log-wrap">
      <div class="log-header">
        <div class="title">Transcript</div>
        <div class="actions">
          <span id="toggleTranscript" class="toggle">Hide</span>
        </div>
      </div>
      <div id="log" class="log"></div>
    </div>

    <div id="analysisWrap" class="analysis-wrap" style="display: none;">
      <div class="log-header">
        <div class="title">Analysis</div>
        <div class="actions">
          <span id="toggleAnalysis" class="toggle">Hide</span>
        </div>
      </div>
      <div id="analysisContent" class="analysis-content"></div>
    </div>

    <audio id="remoteAudio" autoplay></audio>
  </div>

  <script>
    const MODEL = "{MODEL}";
    const BOTS = {BOTS_JSON};

    const scenarioSelect = document.getElementById("scenario");
    const connectBtn = document.getElementById("connect");
    const disconnectBtn = document.getElementById("disconnect");
    const nudgeBtn = document.getElementById("nudge");
    const clearBtn = document.getElementById("clear");
    const analyzeBtn = document.getElementById("analyze");
    const downloadBtn = document.getElementById("download");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const toggleTranscript = document.getElementById("toggleTranscript");
    const toggleAnalysis = document.getElementById("toggleAnalysis");
    const logEl = document.getElementById("log");
    const analysisWrap = document.getElementById("analysisWrap");
    const analysisContent = document.getElementById("analysisContent");
    const remoteAudio = document.getElementById("remoteAudio");
	// Optional: keep partial transcripts if you ever want streaming text later
	const assistantTranscripts = {};
	const userTranscripts = {};

	function handleOaiEvent(ev) {
 	 let msg;
 	 try {
    msg = JSON.parse(ev.data);
 	 } catch {
    return; // ignore non-JSON frames
 	 }
 	 if (!msg || !msg.type) return;

	  // --- Assistant transcript ---
 	 if (msg.type === "response.audio_transcript.delta") {
 	   // Build up partial transcript per response
 	   const id = msg.response_id || "default";
 	   assistantTranscripts[id] = (assistantTranscripts[id] || "") + (msg.delta || "");
 	   return;
 	 }

  	if (msg.type === "response.audio_transcript.done") {
  	  const id = msg.response_id || "default";
 	  const text = (msg.transcript || assistantTranscripts[id] || "").trim();
    	if (text) append("agent", text);
    	delete assistantTranscripts[id];
    	return;
  	}

  	// --- User transcript (Whisper) ---
  	// When input_audio_transcription is enabled, you'll get one of these:
  	if (msg.type === "conversation.item.input_audio_transcription.completed" ||
    	  msg.type === "input_audio_buffer.transcription.completed" ||
    	  msg.type === "conversation.item.input_audio_transcription.delta") {
   	 const text = (msg.transcript || msg.delta || "").trim();
   	 if (text && msg.type === "conversation.item.input_audio_transcription.completed") {
   	   append("user", text);
   	 }
   	 return;
 	 }

  // Log unknown types for debugging:
  console.log("RT event:", msg.type, msg);
}
    // Populate scenarios
    let selectedBotId = BOTS[0]?.id;
    for (const bot of BOTS) {
      const opt = document.createElement("option");
      opt.value = bot.id;
      opt.textContent = bot.label;
      scenarioSelect.appendChild(opt);
    }
    scenarioSelect.value = selectedBotId;
    scenarioSelect.addEventListener("change", () => {
      selectedBotId = scenarioSelect.value;
      append("agent", "Switched to scenario: " + scenarioSelect.options[scenarioSelect.selectedIndex].text);
    });

    let pc = null;
    let micStream = null;

    function setStatus(mode, text) {
      statusText.textContent = text;
      if (mode === "live") {
        statusDot.classList.add("live");
      } else {
        statusDot.classList.remove("live");
      }
    }

    function append(role, text, durMs) {
      const div = document.createElement("div");
      div.className = "msg " + (role === "user" ? "user" : "agent");
      const label = role === "user" ? "You" : "Agent";
      let html = '<span class="label">' + label + ":</span> " + text;
      if (durMs != null) {
        const secs = (durMs / 1000).toFixed(1);
        html += ' <span class="meta">(' + secs + "s)</span>";
      }
      div.innerHTML = html;
      logEl.appendChild(div);
      logEl.scrollTop = logEl.scrollHeight;
    }

    function clearLog() {
      logEl.innerHTML = "";
    }

    // Toggle transcript visibility
    let transcriptVisible = true;
    toggleTranscript.addEventListener("click", () => {
      transcriptVisible = !transcriptVisible;
      if (transcriptVisible) {
        logEl.classList.remove("hidden");
        toggleTranscript.textContent = "Hide";
      } else {
        logEl.classList.add("hidden");
        toggleTranscript.textContent = "Show";
      }
    });

    // Download transcript to .txt
    downloadBtn.addEventListener("click", () => {
      const msgs = Array.from(logEl.querySelectorAll(".msg")).map(div => div.textContent.trim());
      if (!msgs.length) {
        alert("Transcript is empty.");
        return;
      }
      const content = msgs.join("\n");
      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      a.href = url;
      a.download = "transcript-" + ts + ".txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    // Analyze transcript
    analyzeBtn.addEventListener("click", async () => {
      const msgs = Array.from(logEl.querySelectorAll(".msg")).map(div => {
        const text = div.textContent.trim();
        return text;
      });
      if (!msgs.length) {
        alert("Transcript is empty. Please have a conversation first.");
        return;
      }
      const transcript = msgs.join("\n");
      
      analyzeBtn.disabled = true;
      analysisContent.innerHTML = '<div class="analysis-loading">Analyzing transcript...</div>';
      analysisWrap.style.display = "flex";
      // Scroll the analysis panel into view
      setTimeout(() => analysisWrap.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);
      
      try {
        const res = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transcript })
        });
        const data = await res.json();
        
        if (!res.ok) {
          throw new Error(data.error || "Analysis failed");
        }
        
        displayAnalysis(data.analysis);
        // Scroll into view again after results are displayed
        setTimeout(() => analysisWrap.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);
      } catch (err) {
        analysisContent.innerHTML = '<div class="analysis-error">Error: ' + err.message + '</div>';
      } finally {
        analyzeBtn.disabled = false;
      }
    });

    function displayAnalysis(analysis) {
      let html = "";
      
      if (analysis.raw) {
        // Fallback: display raw text if JSON parsing failed
        html = '<div class="analysis-section"><pre style="white-space: pre-wrap;">' + 
               escapeHtml(analysis.raw) + '</pre></div>';
      } else {
        // Display structured analysis
        if (analysis.accuracy) {
          html += '<div class="analysis-section">';
          html += '<h3>Accuracy';
          if (analysis.accuracy.score != null) {
            html += '<span class="score">' + analysis.accuracy.score + '/100</span>';
          }
          html += '</h3>';
          if (analysis.accuracy.overall_assessment) {
            html += '<p>' + escapeHtml(analysis.accuracy.overall_assessment) + '</p>';
          }
          
          // Always show grammar errors section
          html += '<p><strong>Grammar Errors:</strong></p>';
          if (analysis.accuracy.grammar_errors && Array.isArray(analysis.accuracy.grammar_errors) && analysis.accuracy.grammar_errors.length > 0) {
            html += '<ul>';
            analysis.accuracy.grammar_errors.forEach(err => {
              html += '<li>' + escapeHtml(String(err)) + '</li>';
            });
            html += '</ul>';
          } else {
            html += '<p style="color: #9ca3af; font-style: italic;">No grammar errors detected.</p>';
          }
          
          // Always show vocabulary errors section
          html += '<p><strong>Vocabulary Errors:</strong></p>';
          if (analysis.accuracy.vocabulary_errors && Array.isArray(analysis.accuracy.vocabulary_errors) && analysis.accuracy.vocabulary_errors.length > 0) {
            html += '<ul>';
            analysis.accuracy.vocabulary_errors.forEach(err => {
              html += '<li>' + escapeHtml(String(err)) + '</li>';
            });
            html += '</ul>';
          } else {
            html += '<p style="color: #9ca3af; font-style: italic;">No vocabulary errors detected.</p>';
          }
          
          html += '</div>';
        }
        
        if (analysis.complexity) {
          html += '<div class="analysis-section">';
          html += '<h3>Language Complexity</h3>';
          
          // Check if any complexity metrics exist
          const hasComplexity = analysis.complexity.ttr != null || 
                                analysis.complexity.lexical_density != null ||
                                analysis.complexity.mean_tunit_length != null ||
                                analysis.complexity.clauses_per_tunit != null;
          
          if (!hasComplexity) {
            html += '<p style="color: #9ca3af; font-style: italic;">Complexity metrics could not be calculated. Check server logs for details.</p>';
          } else {
            if (analysis.complexity.ttr != null && analysis.complexity.ttr !== undefined) {
              const ttr = typeof analysis.complexity.ttr === 'number' ? analysis.complexity.ttr.toFixed(3) : analysis.complexity.ttr;
              html += '<p><strong>Type-Token Ratio (TTR):</strong> ' + escapeHtml(String(ttr)) + '</p>';
            }
            
            if (analysis.complexity.lexical_density != null && analysis.complexity.lexical_density !== undefined) {
              const ld = typeof analysis.complexity.lexical_density === 'number' ? analysis.complexity.lexical_density.toFixed(3) : analysis.complexity.lexical_density;
              html += '<p><strong>Lexical Density:</strong> ' + escapeHtml(String(ld)) + '</p>';
            }
            
            if (analysis.complexity.mean_tunit_length != null && analysis.complexity.mean_tunit_length !== undefined) {
              const mtl = typeof analysis.complexity.mean_tunit_length === 'number' ? analysis.complexity.mean_tunit_length.toFixed(2) : analysis.complexity.mean_tunit_length;
              html += '<p><strong>Mean Length of T-unit:</strong> ' + escapeHtml(String(mtl)) + ' words</p>';
            }
            
            if (analysis.complexity.clauses_per_tunit != null && analysis.complexity.clauses_per_tunit !== undefined) {
              const cpt = typeof analysis.complexity.clauses_per_tunit === 'number' ? analysis.complexity.clauses_per_tunit.toFixed(2) : analysis.complexity.clauses_per_tunit;
              html += '<p><strong>Clauses per T-unit:</strong> ' + escapeHtml(String(cpt)) + '</p>';
            }
          }
          
          html += '</div>';
        } else {
          // Show complexity section even if empty, to indicate it should be there
          html += '<div class="analysis-section">';
          html += '<h3>Language Complexity</h3>';
          html += '<p style="color: #9ca3af; font-style: italic;">Complexity metrics not available. Check server logs for details.</p>';
          html += '</div>';
        }
      }
      
      analysisContent.innerHTML = html || '<div class="analysis-loading">No analysis data available.</div>';
      analysisContent.scrollTop = 0;
      // Ensure the analysis panel is visible
      analysisWrap.style.display = "flex";
    }

    function escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text;
      return div.innerHTML;
    }

    // Toggle analysis visibility
    let analysisVisible = true;
    toggleAnalysis.addEventListener("click", () => {
      analysisVisible = !analysisVisible;
      if (analysisVisible) {
        analysisContent.style.display = "block";
        toggleAnalysis.textContent = "Hide";
      } else {
        analysisContent.style.display = "none";
        toggleAnalysis.textContent = "Show";
      }
    });

    function REALTIME_URL() {
      const base = "{REALTIME_URL_PLACEHOLDER}";
      const url = base + "?model=" + encodeURIComponent(MODEL);
      return url;
    }

    async function waitForIceGathering(pc) {
      if (pc.iceGatheringState === "complete") {
        return;
      }
      await new Promise((resolve) => {
        const checkState = () => {
          if (pc.iceGatheringState === "complete") {
            pc.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        };
        pc.addEventListener("icegatheringstatechange", checkState);
        checkState();
      });
    }

    async function connect() {
      connectBtn.disabled = true;
      try {
        setStatus("idle", "requesting session...");
        // 1) Ask server for session (returns ephemeral key and config)
        const sessRes = await fetch("/session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ bot_id: selectedBotId })
        });
        const session = await sessRes.json();
        if (!sessRes.ok) {
          throw new Error(session.error || "Failed to create session");
        }

        // 2) Mic
        setStatus("idle", "requesting microphone...");
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // 3) WebRTC peer connection
        pc = new RTCPeerConnection();
        let dataChannel = null;
        
        // OpenAI creates the data channel, so we listen for it
        pc.ondatachannel = (event) => {
          const channel = event.channel;
          console.log("Data channel received:", channel.label);
          if (channel.label === "oai-events") {
            dataChannel = channel;
            channel.onopen = () => {
              console.log("oai-events channel open");
              // Send session update to enable audio and text modalities
              channel.send(JSON.stringify({
                type: "session.update",
                session: {
                  modalities: ["audio", "text"],
                  input_audio_transcription: {
                    model: "whisper-1"
                  }
                }
              }));
            };
            channel.onmessage = handleOaiEvent;
            channel.onerror = (err) => console.warn("oai-events channel error", err);
            channel.onclose = () => console.log("oai-events channel closed");
          }
        };
        pc.addTransceiver("audio", { direction: "sendrecv" });
        pc.ontrack = (e) => {
          if (e.streams && e.streams[0]) {
            remoteAudio.srcObject = e.streams[0];
          }
        };
        for (const track of micStream.getTracks()) {
          pc.addTrack(track, micStream);
        }

        // 4) Create SDP offer and send to OpenAI Realtime
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGathering(pc); // ensure ICE candidates are included in SDP

        setStatus("idle", "starting realtime session...");
        const sdpResponse = await fetch(REALTIME_URL(), {
          method: "POST",
          body: pc.localDescription.sdp,
          headers: {
            Authorization: "Bearer " + session.client_secret.value,
            "Content-Type": "application/sdp",
            "OpenAI-Beta": "realtime=v1"
          }
        });
        const sdpText = await sdpResponse.text();
        if (!sdpResponse.ok) {
          append("agent", "Realtime handshake failed: " + sdpText);
          throw new Error("Handshake failed");
        }

        const answer = { type: "answer", sdp: sdpText };
        await pc.setRemoteDescription(answer);

        // Wait a bit for the data channel to open
        await new Promise(resolve => setTimeout(resolve, 500));

        setStatus("live", "connected");
        connectBtn.disabled = true;
        disconnectBtn.disabled = false;
        nudgeBtn.disabled = false;
        append("agent", "Connected. Speak whenever you like.");

      } catch (err) {
        console.error(err);
        setStatus("idle", "idle");
        connectBtn.disabled = false;
        disconnectBtn.disabled = true;
        nudgeBtn.disabled = true;
        append("agent", "Connect error: " + err.message);
      }
    }

    function disconnect() {
      if (pc) {
        pc.close();
        pc = null;
      }
      if (micStream) {
        for (const t of micStream.getTracks()) t.stop();
        micStream = null;
      }
      setStatus("idle", "idle");
      connectBtn.disabled = false;
      disconnectBtn.disabled = true;
      nudgeBtn.disabled = true;
      append("agent", "Disconnected.");
    }

    // Simple "nudge" – mainly a UI affordance for students
    async function nudge() {
      if (!pc || !micStream) return;
      append("user", "🎙️ (speaking)");
      // With server VAD, just speaking is enough; no explicit audio event needed here.
    }

    clearBtn.addEventListener("click", clearLog);
    connectBtn.addEventListener("click", () => { connect(); });
    disconnectBtn.addEventListener("click", () => { disconnect(); });
    nudgeBtn.addEventListener("click", () => { nudge(); });

    // Initial status
    setStatus("idle", "idle");
    append("agent", "Choose a scenario and click Connect to begin.");
  </script>
</body>
</html>
"""

# Inject realtime URL constant
REALTIME_HTML = REALTIME_HTML.replace("{REALTIME_URL_PLACEHOLDER}", REALTIME_URL)


if __name__ == "__main__":
    # Support both local and cloud deployment
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    app.run(host=host, port=port, debug=debug)
