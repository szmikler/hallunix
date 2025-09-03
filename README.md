> Vibe-coded

# Hallunix

**Hallunix** is an AI-simulated Unix-like environment.  
It mimics a Linux shell/OS by routing all commands through an LLM via [LiteLLM](https://github.com/BerriAI/litellm).  

Hallunix starts with a neofetch-style logo, provides a prompt (`user@hallunix:~$ `), and responds to commands realistically — including interactive interpreters like Python. Session history is ephemeral (per run) and no real commands are executed on your machine.

---

## Features

- **LLM-driven OS simulation**: every command is interpreted and answered by a large language model.
- **Authentic UX**:  
  - Startup neofetch logo with system details.  
  - Prompts (`user@hallunix:~$ `, `>>> `) controlled by the model.  
  - Prompts end with a single space and cannot be backspaced into (protected by `readline`).  
- **Interactive interpreters**: switch into Python (`python`) or similar REPLs naturally.
- **History command**: `:history` shows all inputs from the current session.
- **Ephemeral state**: no persistence across runs.
- **Streaming output** (optional): stream results as they arrive, or disable with `--no-stream`.

---

```
usage: hallunix [-h] [--model MODEL] [--context-turns N] [--no-stream]
                [--temperature FLOAT] [--max-output-chars N]

AI-simulated Unix-like environment over LiteLLM

options:
  --model MODEL             Model to use (default: gpt-4o-mini)
  --context-turns N         Number of past turns to keep in context (default: 20)
  --no-stream               Disable streaming responses
  --temperature FLOAT       Sampling temperature (default: None)
  --max-output-chars N      Max characters before truncating output
```
