# Chapter 30: Voice and Multimodal Agents
## STT, TTS, and the Realtime Gap

**From:** *Design of Agentic Systems with Case Studies*
**Chapter type:** Type A — Architectural Pattern
**Course:** INFO 7375: Prompt Engineering for Generative AI

---

## The Architectural Argument

Realtime voice (speech-to-speech) looks better in demos than it performs in production. The STT→LLM→TTS pipeline dominates enterprise deployment for one architectural reason: it externalizes the voice activity detection (VAD) boundary decision, making it observable, logged, and recoverable. Speech-to-speech internalizes that decision inside the model's forward pass, where it is invisible to every recovery mechanism.

**The book's master claim:** Architecture is the leverage point, not the model.

**This chapter's instance:** VAD failure is an architectural failure. The pipeline makes it observable. Speech-to-speech does not. The same VAD mis-segmentation that is silently unrecoverable in a speech-to-speech system produces a logged, thresholdable, escalatable event in a pipeline — without any change to the LLM, the STT engine, or the VAD module itself.

---

## The Failure Mode

A patient calls a hospital triage line and says: *"I have a fever and my—"*

The VAD fires early. The audio chunk is closed mid-sentence. The STT engine transcribes it accurately: *"I have a fever and my."* The LLM receives a well-formed string, has no signal that it is incomplete, and generates a plausible clinical response. No stage reported an error. The failure is architectural, not computational.

**Notebook confirmation (confirmed run):**

| Utterance | VAD confidence | Tokens | Closure | Gate decision |
|---|---|---|---|---|
| Full (2.9s) | 0.990 | 13 | True | FORWARD |
| Truncated (2.05s) | 0.344 | 6 | False | ESCALATE |

Confidence delta: **0.646** — the observable signal the recovery layer acts on.

---
`
## Repository Structure

```
chapter30-voice-multimodal-agents/
├── README.md                        ← this file
├── chapter30.md                     ← full chapter prose (Sections 1–5 + images)
├── authors_note.md                  ← 3-page Author's Note (design choices, tool usage, self-assessment)
├── chapter30_notebook.py         ← runnable demo (VAD failure + defense architecture)
├── requirements.txt                 ← dependencies
└── images/
    ├── figure1_topology.png         ← pipeline vs. speech-to-speech topology
    ├── figure2_causal_chain.png     ← VAD failure causal chain
    └── figure3_recovery_gate.png    ← recovery gate decision logic
```

---

## Quickstart

### Requirements

- Python 3.10+
- OpenAI API key

### Install

```bash
pip install -r requirements.txt
```

### Set API key

```bash
export OPENAI_API_KEY=your_key_here
```

### Run

```bash
python chapter30_notebook.py
```

Or open as a Jupyter notebook and run cells top to bottom.

---

## What the Demo Does

The notebook has seven parts:

**Part 1 — Audio simulation.** Builds two utterances as numpy arrays: a full utterance (2.9s with 0.4s trailing silence) and a truncated utterance (2.05s with 0.05s trailing silence). No microphone required — the failure is deterministic and reproducible from a fresh clone.

**Part 2 — VAD gate.** Runs both utterances through a VAD module that produces a logged confidence score on every decision. The full utterance scores 0.990 (FORWARD). The truncated utterance scores 0.344 (HOLD\_FOR\_REVIEW). The confidence delta of 0.646 is the observable signal.

**Part 3 — STT stage.** Simulates transcription. The full utterance produces a 13-token transcript with syntactic closure. The truncated utterance produces a 6-token transcript with no closure. Both are accurate representations of their audio input. No stage has made an error yet.

**Part 4 — LLM stage.** Sends both transcripts to the LLM with a clinical triage system prompt. The full transcript receives an appropriate emergency response. The truncated transcript receives a plausible response to a sentence the patient never finished. No stage reports an error.

**Part 5 — Mandatory Human Decision Node.** Execution halts. The student must document their threshold decision before proceeding. Default thresholds: VAD confidence ≥ 0.75, minimum 8 tokens, syntactic closure required.

**Part 6 — Defense architecture.** Runs the defended pipeline on both utterances. The recovery gate checks three signals — VAD confidence, token count, syntactic closure — before invoking the LLM. The truncated utterance triggers ESCALATE on all three checks. The LLM is not invoked.

**Part 7 — Reader exercise.** Parameterized by `truncation_s`. Change this value and observe how VAD confidence, gate decision, and LLM output change across 1.5, 2.0, 2.5, and 3.0 seconds.

---

## The Mandatory Human Decision Node

Located at Cell 8 (Part 5). Full stop before the defense architecture runs.

```python
# MANDATORY HUMAN DECISION NODE
# The architecture assumes that realtime processing
# can reliably detect speech boundaries using VAD.
#
# Before proceeding:
# Verify whether the default thresholds hold for your environment.
#
# VAD confidence threshold:  0.75
# Minimum token count:       8
# Require syntactic closure: True
#
# Document your verification or rejection below:
```

The default threshold of 0.75 was accepted for this demo after observing a confidence delta of 0.646 between the full and truncated utterances. In a production ICU deployment, this threshold requires benchmarking against actual background noise profiles and population-specific pause duration distributions.

---

## The Reader Exercise

Open `chapter30_notebook.py` at Cell 11. Change only this line:

```python
truncation_s = 2.0  # <-- CHANGE THIS VALUE
```

Try: `1.5`, `2.0`, `2.5`, `3.0`

Record for each value: VAD confidence score, gate decision (ESCALATE or FORWARD), LLM response text.

**The question to answer:** At what truncation value does the gate transition from predominantly ESCALATE to predominantly FORWARD? That transition point is the architectural vulnerability — not a model failure, not a VAD failure, but the boundary of the recovery layer's detection capability.

---

## AI Tool Usage

This chapter was produced using the following tools as co-authors:

| Tool | Role | Where used |
|---|---|---|
| Bookie the Bookmaker | Prose generation | Sections 2–4 drafted from scenario + mechanism prompts |
| Figure Architect | Figure prompts | Three figures generated from stable draft |
| Eddy the Editor | Audit | 11 findings across 4 lenses; 9 accepted, 2 modified |

Human Decision Nodes and AI rejections are documented in `authors_note.md` Page 2.

---

## Video

*[Insert YouTube/Vimeo link here]*

Structure: Explain (2–3 min) → Show (5–6 min) → Try (2–3 min)

The Human Decision Node moment occurs at 6:30 — Cell 8, on camera:
*"The AI proposed a VAD threshold of 0.75 without knowing my acoustic environment. I accepted it after observing a confidence delta of 0.646 in the notebook output — but in a production ICU deployment I would benchmark this against real noise profiles before accepting any fixed threshold."*

---

## Key Takeaway

Choose the pipeline whenever a failure requires an explanation.
Choose speech-to-speech only when a failure requires nothing more than a retry.
