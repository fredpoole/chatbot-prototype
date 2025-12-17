import os
import json
import sys
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

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
# Enable CORS for all routes
CORS(app, origins=["http://localhost:7050", "http://127.0.0.1:7050"])


@app.route("/bots", methods=["GET"])
def get_bots():
    """Return list of available bots."""
    return jsonify(BOTS)


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
        return jsonify({"error": f"Failed to create session: {e}"}), 500

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
            return jsonify({"error": "No transcript provided"}), 400
        
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
            return jsonify({
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
            }), 200
        
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
        
        return jsonify({"analysis": combined_analysis}), 200
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API request failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500


if __name__ == "__main__":
    # Backend runs on port 5050
    port = int(os.getenv("PORT", 5050))
    host = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    print(f"Starting backend server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
