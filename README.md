# HallUnix

HallUnix is a fully hallucinated Unix-like OS environment.

It mimics a Linux shell/OS by routing all commands through an LLM via [LiteLLM](https://github.com/BerriAI/litellm).  

HallUnix starts with a fresh OS session and responds to commands realistically — including interactive interpreters like Python. Session history is ephemeral (per run) and no real commands are executed on your machine.

## Features

- **Mostly vibe-coded**: which means it might not work
- **LLM-driven OS simulation**: every command is interpreted and answered by LLM
- **Authentic UX**: startup neofetch logo with system details
- **Interactive interpreters**: work in Python (`python`) or similar REPLs naturally
- **Ephemeral state**: no persistence across runs
- **Command history**: typing `::history` shows all commands from the session (why? idk)
- **Possible to exit**: by typing `::exit`

## How to use

Use with your own API keys and your favourite LLM.

```
export OPENAI_API_KEY="sk-..."
python hallunix.py --model openai/gpt-5
```
