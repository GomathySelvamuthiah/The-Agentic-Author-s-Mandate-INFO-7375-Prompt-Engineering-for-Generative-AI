from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# %% Cell 1 — Imports and config
import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

SAMPLE_RATE = 16_000  # Hz — standard for speech models
SILENCE_THRESHOLD = 0.01  # RMS energy below this = silence
VAD_CONFIDENCE_THRESHOLD = 0.75  # gate: below this, hold for review

# %% [markdown]
# ## Part 1 — Simulated Audio and Utterance Pairs
#
# We simulate audio as numpy arrays rather than recording live audio.
# This makes the failure reproducible from a fresh clone without a microphone.
# The simulation preserves the acoustic property that matters:
# energy envelope shape at the utterance boundary.

# %% Cell 2 — Audio simulation helpers
def make_speech_segment(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate a speech segment: shaped noise with non-zero energy."""
    samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, samples)
    # Shaped noise approximating speech energy envelope
    envelope = np.sin(np.pi * t / duration_s) ** 0.5
    noise = np.random.randn(samples) * 0.3
    return (envelope * noise).astype(np.float32)

def make_silence_segment(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate a silence segment: near-zero energy with floor noise."""
    samples = int(duration_s * sr)
    return (np.random.randn(samples) * 0.005).astype(np.float32)

def rms_energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))

def build_utterance(speech_s: float, trailing_silence_s: float) -> np.ndarray:
    """Build an utterance: speech followed by trailing silence."""
    return np.concatenate([
        make_speech_segment(speech_s),
        make_silence_segment(trailing_silence_s)
    ])

# Full utterance: speech + natural trailing silence (~400ms)
FULL_UTTERANCE = build_utterance(speech_s=2.5, trailing_silence_s=0.4)

# Truncated utterance: speech cut at 2s — no trailing silence, mid-energy cut
TRUNCATED_UTTERANCE = build_utterance(speech_s=2.0, trailing_silence_s=0.05)

print(f"Full utterance:      {len(FULL_UTTERANCE)/SAMPLE_RATE:.2f}s  "
      f"RMS={rms_energy(FULL_UTTERANCE):.4f}")
print(f"Truncated utterance: {len(TRUNCATED_UTTERANCE)/SAMPLE_RATE:.2f}s  "
      f"RMS={rms_energy(TRUNCATED_UTTERANCE):.4f}")

# %% [markdown]
# ## Part 2 — VAD Module (Observable Gate)
#
# In the STT→LLM→TTS pipeline, VAD is a discrete, inspectable gate.
# Every decision is logged with a confidence score.
# This is the architectural property that makes recovery possible.

# %% Cell 3 — VAD implementation
@dataclass
class VADResult:
    accepted: bool
    confidence: float
    trailing_silence_s: float
    audio_duration_s: float
    log_entry: dict = field(default_factory=dict)

def vad_gate(audio: np.ndarray, sr: int = SAMPLE_RATE) -> VADResult:
    """
    Simulated VAD gate. Returns a confidence score based on:
    - Trailing silence duration (longer = more likely complete utterance)
    - Energy profile at the boundary (drop = more likely complete)
    
    In production: replace with WebRTC VAD, Silero VAD, or equivalent.
    The interface (confidence score, logged event) remains identical.
    """
    duration_s = len(audio) / sr

    # Measure trailing 300ms energy
    tail_samples = int(0.3 * sr)
    tail_energy = rms_energy(audio[-tail_samples:])
    body_energy = rms_energy(audio[:-tail_samples]) if len(audio) > tail_samples else 1.0

    # Trailing silence ratio: how much quieter is the tail vs the body?
    silence_ratio = 1.0 - min(tail_energy / (body_energy + 1e-9), 1.0)

    # Estimate trailing silence duration
    # Walk back from end to find where energy rises above floor
    window = int(0.05 * sr)  # 50ms windows
    trailing_silence_s = 0.0
    for i in range(len(audio) // window, 0, -1):
        chunk = audio[(i-1)*window : i*window]
        if rms_energy(chunk) > SILENCE_THRESHOLD:
            break
        trailing_silence_s += 0.05

    # Confidence: weighted combination of silence ratio and trailing silence
    confidence = min(0.5 * silence_ratio + 0.5 * min(trailing_silence_s / 0.3, 1.0), 1.0)
    accepted = confidence >= VAD_CONFIDENCE_THRESHOLD

    log_entry = {
        "timestamp": time.time(),
        "audio_duration_s": round(duration_s, 3),
        "trailing_silence_s": round(trailing_silence_s, 3),
        "tail_energy": round(tail_energy, 5),
        "body_energy": round(body_energy, 5),
        "confidence": round(confidence, 4),
        "accepted": accepted,
        "action": "FORWARD" if accepted else "HOLD_FOR_REVIEW"
    }

    return VADResult(accepted, confidence, trailing_silence_s, duration_s, log_entry)

# %% Cell 4 — Run VAD on both utterances
print("=" * 60)
print("VAD GATE — FULL UTTERANCE")
full_vad = vad_gate(FULL_UTTERANCE)
print(json.dumps(full_vad.log_entry, indent=2))

print("\n" + "=" * 60)
print("VAD GATE — TRUNCATED UTTERANCE")
trunc_vad = vad_gate(TRUNCATED_UTTERANCE)
print(json.dumps(trunc_vad.log_entry, indent=2))

print("\n" + "=" * 60)
print(f"Confidence delta: {full_vad.confidence - trunc_vad.confidence:.4f}")
print("This delta is the observable signal the recovery layer acts on.")

# %% [markdown]
# ## Part 3 — STT Stage (Simulated Transcripts)
#
# We simulate STT output rather than calling a live API to keep the
# failure deterministic and reproducible without audio hardware.
# In production, replace simulate_stt() with a Whisper API call.

# %% Cell 5 — Simulated STT
@dataclass
class STTResult:
    transcript: str
    token_count: int
    has_syntactic_closure: bool  # ends with . ? ! or complete clause

def simulate_stt(audio: np.ndarray, sr: int = SAMPLE_RATE) -> STTResult:
    """
    Simulates STT output based on audio duration.
    Shorter audio → truncated transcript.
    In production: replace with client.audio.transcriptions.create()
    """
    duration_s = len(audio) / sr

    if duration_s >= 2.8:
        # Full utterance
        transcript = "I have a fever and my chest has been hurting since this morning."
        closure = True
    else:
        # Truncated — mid-sentence, no closure
        transcript = "I have a fever and my"
        closure = False

    tokens = transcript.split()
    return STTResult(
        transcript=transcript,
        token_count=len(tokens),
        has_syntactic_closure=closure
    )

# %% Cell 6 — Run STT on both utterances
print("STT — FULL UTTERANCE")
full_stt = simulate_stt(FULL_UTTERANCE)
print(f"  Transcript:        '{full_stt.transcript}'")
print(f"  Token count:       {full_stt.token_count}")
print(f"  Syntactic closure: {full_stt.has_syntactic_closure}")

print("\nSTT — TRUNCATED UTTERANCE")
trunc_stt = simulate_stt(TRUNCATED_UTTERANCE)
print(f"  Transcript:        '{trunc_stt.transcript}'")
print(f"  Token count:       {trunc_stt.token_count}")
print(f"  Syntactic closure: {trunc_stt.has_syntactic_closure}")

# %% [markdown]
# ## Part 4 — LLM Stage
#
# The LLM receives the transcript and generates a clinical response.
# It has no access to the audio, no VAD confidence score, no signal
# that the utterance was truncated. It behaves correctly given its input.
# This is the architectural argument: the model is not broken. The boundary is.

# %% Cell 7 — LLM call
SYSTEM_PROMPT = """You are a triage nurse assistant handling overnight calls.
A patient is describing their symptoms. Respond with an initial assessment
and next steps. Be specific and clinical. Keep response under 60 words."""

def call_llm(transcript: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ],
        max_tokens=120,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

print("LLM — FULL UTTERANCE TRANSCRIPT")
full_response = call_llm(full_stt.transcript)
print(f"  Input:    '{full_stt.transcript}'")
print(f"  Response: {full_response}")

print("\nLLM — TRUNCATED UTTERANCE TRANSCRIPT")
trunc_response = call_llm(trunc_stt.transcript)
print(f"  Input:    '{trunc_stt.transcript}'")
print(f"  Response: {trunc_response}")

print("\n" + "=" * 60)
print("OBSERVE: The LLM generated a clinical response to an incomplete sentence.")
print("No stage reported an error. The failure is architectural, not computational.")

# %% [markdown]
# ## Part 5 — MANDATORY HUMAN DECISION NODE
#
# ============================================================
# STOP. READ THIS BEFORE PROCEEDING.
# ============================================================
#
# The VAD gate above uses a confidence threshold of 0.75.
# The recovery layer below uses a minimum token count of 8
# and requires syntactic closure before forwarding to the LLM.
#
# These thresholds encode an architectural assumption:
#   A response held for review costs less than a wrong response delivered.
#
# Before accepting these defaults, verify:
#   1. Is a 0.75 VAD confidence threshold appropriate for your
#      acoustic environment? (ICU background noise will shift this.)
#   2. Is a minimum 8-token transcript the right completeness signal
#      for your domain? (Medical sentences are often longer.)
#   3. Does your deployment context prefer false positives (holding
#      complete utterances) or false negatives (passing truncated ones)?
#
# Document your decision and reasoning below before running Part 6.
#
# YOUR DECISION:
# [ ] I accept the default thresholds for the purposes of this demo.
# [ ] I am modifying the thresholds. My reasoning:
#     _______________________________________________
#
# ============================================================

# %% Cell 8 — Human Decision Node — set thresholds here
# Modify these values based on your documented decision above.
RECOVERY_MIN_TOKENS = 8          # minimum tokens for a complete utterance
RECOVERY_REQUIRE_CLOSURE = True  # require syntactic closure marker

print("Human Decision Node acknowledged.")
print(f"  VAD confidence threshold:    {VAD_CONFIDENCE_THRESHOLD}")
print(f"  Minimum token count:         {RECOVERY_MIN_TOKENS}")
print(f"  Require syntactic closure:   {RECOVERY_REQUIRE_CLOSURE}")
print("\nProceed to Part 6 only after documenting your threshold decision.")

# %% [markdown]
# ## Part 6 — Defense Architecture
#
# The recovery layer operates at the pipeline boundary — between STT and LLM.
# It does not modify the LLM. It does not modify the STT engine.
# It adds a gate that checks the observable signals the pipeline already produces:
# VAD confidence, token count, and syntactic closure.
# This is what it means for the architecture to be defensible.

# %% Cell 9 — Recovery layer
@dataclass
class RecoveryDecision:
    action: str  # FORWARD | HOLD | ESCALATE
    reason: str
    vad_confidence: float
    token_count: int
    has_closure: bool

def recovery_gate(
    vad_result: VADResult,
    stt_result: STTResult,
    min_tokens: int = RECOVERY_MIN_TOKENS,
    require_closure: bool = RECOVERY_REQUIRE_CLOSURE
) -> RecoveryDecision:
    """
    Recovery layer: checks three observable signals before forwarding to LLM.
    Operates entirely on pipeline artifacts — no audio, no model internals.
    """
    reasons = []

    if vad_result.confidence < VAD_CONFIDENCE_THRESHOLD:
        reasons.append(f"VAD confidence {vad_result.confidence:.2f} < {VAD_CONFIDENCE_THRESHOLD}")

    if stt_result.token_count < min_tokens:
        reasons.append(f"Token count {stt_result.token_count} < minimum {min_tokens}")

    if require_closure and not stt_result.has_syntactic_closure:
        reasons.append("No syntactic closure marker detected")

    if not reasons:
        action = "FORWARD"
        reason = "All recovery checks passed"
    elif vad_result.confidence < 0.4:
        action = "ESCALATE"
        reason = " | ".join(reasons)
    else:
        action = "HOLD"
        reason = " | ".join(reasons)

    return RecoveryDecision(
        action=action,
        reason=reason,
        vad_confidence=vad_result.confidence,
        token_count=stt_result.token_count,
        has_closure=stt_result.has_syntactic_closure
    )

# %% Cell 10 — Run full defended pipeline on both utterances
def run_defended_pipeline(audio: np.ndarray, label: str):
    print(f"\n{'='*60}")
    print(f"DEFENDED PIPELINE — {label}")
    print(f"{'='*60}")

    vad = vad_gate(audio)
    print(f"VAD:      confidence={vad.confidence:.3f}  action={vad.log_entry['action']}")

    stt = simulate_stt(audio)
    print(f"STT:      '{stt.transcript}'  tokens={stt.token_count}  closure={stt.has_syntactic_closure}")

    decision = recovery_gate(vad, stt)
    print(f"RECOVERY: action={decision.action}  reason='{decision.reason}'")

    if decision.action == "FORWARD":
        response = call_llm(stt.transcript)
        print(f"LLM:      {response}")
    else:
        print(f"LLM:      [NOT INVOKED — utterance held pending review]")
        print(f"          Recovery action: {decision.action}")

run_defended_pipeline(FULL_UTTERANCE, "FULL UTTERANCE")
run_defended_pipeline(TRUNCATED_UTTERANCE, "TRUNCATED UTTERANCE (VAD FAILURE)")

# %% [markdown]
# ## Part 7 — Try It Yourself (The Exercise)
#
# The exercise for the reader. Instructions are in Section 5 of the chapter.
# See the cell below — modify ONLY the truncation point and observe
# what changes at each stage of the pipeline.

# %% Cell 11 — Reader exercise: trigger the failure yourself
# ============================================================
# EXERCISE: Change truncation_s to different values and observe:
#   - How VAD confidence changes
#   - Whether the recovery gate catches the truncation
#   - What the LLM generates when the gate fails to catch it
#
# Start here: try truncation_s = 1.5, 2.0, 2.5, 3.0
# ============================================================

truncation_s = 2.0  # <-- CHANGE THIS VALUE

exercise_audio = build_utterance(speech_s=truncation_s, trailing_silence_s=0.05)
print(f"Exercise: truncation_s={truncation_s}s")
run_defended_pipeline(exercise_audio, f"EXERCISE UTTERANCE ({truncation_s}s)")

# At what truncation_s does the recovery gate start passing truncated utterances?
# That threshold is the architectural vulnerability you are mapping.
