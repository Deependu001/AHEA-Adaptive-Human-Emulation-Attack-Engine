# Adaptive Human-Emulation Attack Engine (AHEA)

AHEA is a safe, research-focused adversary emulator. It simulates attacker behavior (scan → fingerprint → post-ex → exfil), observes defender reactions, learns with reinforcement learning, and retrains an ML predictor using real run data. The engine visualizes evolving attack paths in an attack graph.

---

## ✨ Features
- **Learning engine:** Q-values evolve as the engine runs.
- **Attack graph visualization:** NetworkX + Matplotlib with colored edges (green/yellow/red).
- **CSV logging:** Epoch-by-epoch actions, rewards, defender reactions.
- **ML retraining:** Decision tree retrains on `run_log.csv`.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt