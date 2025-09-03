#!/usr/bin/env python3
"""
Hallunix — hallucinated Unix-like environment

AI-simulated Unix-like OS over LiteLLM. Commands are sent to an LLM which
returns the *terminal output* to print. Session history is in-memory only.

Behavior:
- The LLM controls prompts (e.g., "user@hallunix:~$ ", ">>> "). We do not add our
  own newlines; we preserve the model's output exactly.
- On startup, we show a login header including a neofetch-style logo that ends
  with the prompt (with a space, no trailing newline) so you can type immediately.
- We pass the prompt to `input(prompt)` so readline protects it.
- The same header string is sent as the first assistant message so the model
  continues from that exact state.
- Output must never include Markdown code fences; we strip them defensively.

Requirements:
  - Python 3.9+
  - `pip install litellm`
  - Readline is required; ImportError is not suppressed.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Required deps; let ImportError surface if missing
from litellm import completion
import readline  # noqa: F401

PROMPT_DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o-mini")

# Neofetch-style logo and system info (improved)
NEOFETCH = (
    "       _          _           \n"
    "      | |        (_)          \n"
    "      | |__  _ __ _ _   _ _ __ \n"
    "      | '_ \\| '__| | | | | '_ \\ \n"
    "      | | | | |  | | |_| | | | |\n"
    "      |_| |_|_|  |_|\\__,_|_| |_|   hallunix 1.0\n"
    "\n"
    "OS: Hallunix x86_64\n"
    "Host: hallunix\n"
    "Kernel: 6.16.3\n"
    "Uptime: 0 min\n"
    "Shell: bash 5.2.37\n"
    "CPU: AMD Ryzen Threadripper 7980X 64-Cores (128) @ 5.65 GHz\n"
    "GPU: NVIDIA GeForce RTX 5090 [Discrete]\n"
    "Memory: 8.24 GiB / 125.22 GiB\n"
    "Disk (/): 233.61 GiB / 1.86 TiB (12%) - btrfs\n"
    "Local IP (enp69s0): 192.168.1.27/24\n"
    "Locale: en_US.UTF-8\n"
    "\n"
)

# Header + prompt with NO trailing newline; user can type immediately after "$ "
INIT_HEADER = (
    f"{NEOFETCH}" "user@hallunix:~$ "
)

SYSTEM_PROMPT = (
    "You are hallunix, an AI-simulated Linux/Unix-like OS.\n"
    "Act like a full OS shell: interpret each user line strictly as a command and\n"
    "output ONLY the terminal output (stdout/stderr) for that command.\n"
    "NEVER include explanations, Markdown, code fences, or surrounding quotes.\n"
    "If a command would produce no output, return a blank string. If invalid,\n"
    "return the typical Unix error. Assume a standard GNU userland.\n\n"
    "State & environment: Start in /home/user with a plausible UNIX-like FS.\n"
    "Maintain cwd and simulated file/process state consistently using prior\n"
    "commands+outputs as truth. Do not claim '.' is missing.\n\n"
    "Header & prompts: The session begins with the neofetch-style header below,\n"
    "which already ends with a prompt and no trailing newline. Treat that exact\n"
    "text as your first assistant message and continue from there. When emitting a\n"
    "prompt (shell `$` prompt or Python `>>>`), end it with a single space and do\n"
    "not add a trailing newline. Examples: 'user@hallunix:~$ ', '>>> '.\n\n"
    "Large outputs: be realistic; truncate like a pager/head with ellipses when huge.\n"
)


def strip_code_fences(text: str) -> str:
    """Remove common Markdown fences if a model emits them."""
    if not text:
        return ""
    t = text
    s = t.strip("\n")
    if s.startswith("```") and s.endswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            t = "\n".join(lines[1:-1])
    if t.startswith("```"):
        t = t.lstrip("`")
    if t.endswith("```"):
        t = t.rstrip("`")
    return t


def split_screen_and_prompt(s: str) -> Tuple[str, str]:
    if not s:
        return "", ""
    if s.endswith("\n"):
        return s, ""
    nl = s.rfind("\n")
    if nl == -1:
        return "", s
    return s[: nl + 1], s[nl + 1 :]


@dataclass
class Turn:
    command: str
    output: str


class Hallunix:
    def __init__(
        self,
        model: str,
        stream: bool = True,
        context_turns: int = 20,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_output_chars: int = 20000,
    ) -> None:
        self.model = model
        self.stream = stream
        self.context_turns = max(0, context_turns)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_chars = max_output_chars
        self.history: List[Turn] = []
        self._header_seeded: bool = False
        self._current_prompt: str = ""

    def run(self) -> None:
        header_screen, header_prompt = split_screen_and_prompt(INIT_HEADER)
        print(header_screen, end="")
        self._current_prompt = self._normalize_prompt_for_input(header_prompt)
        self._header_seeded = True

        while True:
            try:
                line = input(self._current_prompt)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

            if not line.strip():
                out, _ = self._exec_with_llm("")
                screen, prompt = split_screen_and_prompt(out)
                if screen:
                    print(screen, end="")
                if prompt:
                    self._current_prompt = self._normalize_prompt_for_input(prompt)
                continue

            if line.strip() in {"exit", "quit", ":q", "logout"}:
                break
            if line.strip() == ":history":
                self._print_history()
                continue

            output, _ = self._exec_with_llm(line)
            if output is not None:
                screen, prompt = split_screen_and_prompt(output)
                if screen:
                    print(screen, end="")
                if prompt:
                    self._current_prompt = self._normalize_prompt_for_input(prompt)
            self.history.append(Turn(command=line, output=output or ""))

    def _build_messages(self, new_command: str) -> list:
        messages = [{"role": "system", "content": self.system_prompt}]
        if self._header_seeded:
            messages.append({"role": "assistant", "content": INIT_HEADER})
        if self.history and self.context_turns:
            recent = self.history[-self.context_turns :]
            for t in recent:
                messages.append({"role": "user", "content": t.command})
                messages.append({"role": "assistant", "content": t.output or ""})
        messages.append({"role": "user", "content": new_command})
        return messages

    def _exec_with_llm(self, command: str) -> Tuple[str, bool]:
        messages = self._build_messages(command)
        try:
            kwargs = {"model": self.model, "messages": messages, "stream": self.stream}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            acc: List[str] = []
            if self.stream:
                for chunk in completion(**kwargs):
                    delta = (
                        getattr(chunk.choices[0].delta, "content", None)
                        if hasattr(chunk.choices[0], "delta")
                        else chunk.choices[0]["delta"].get("content")
                    )
                    if not delta:
                        continue
                    acc.append(delta)
                    if sum(len(x) for x in acc) >= self.max_output_chars:
                        acc.append("\n[hallunix: output truncated]\n")
                        break
                content = "".join(acc)
            else:
                resp = completion(**kwargs)
                content = resp["choices"][0]["message"]["content"] or ""
            cleaned = strip_code_fences(content)
            return cleaned, False
        except KeyboardInterrupt:
            return "^C", False
        except Exception as e:
            return f"hallunix: error contacting LLM: {type(e).__name__}: {e}", False

    def _normalize_prompt_for_input(self, prompt: str) -> str:
        if not prompt:
            return ""
        return prompt.rstrip() + " "

    def _print_history(self) -> None:
        if not self.history:
            print("(no history)")
            return
        width = len(str(len(self.history)))
        for i, t in enumerate(self.history, start=1):
            num = str(i).rjust(width)
            print(f"{num}  {t.command}")


def _positive_int(val: str) -> int:
    try:
        iv = int(val)
        if iv < 0:
            raise ValueError
        return iv
    except Exception:
        raise argparse.ArgumentTypeError("must be non-negative integer")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hallunix",
        description="AI-simulated Unix-like environment over LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              hallunix --model gpt-4o-mini
              hallunix --model openrouter/anthropic/claude-3.5-sonnet
              hallunix --context-turns 50 --no-stream
            """
        ),
    )
    p.add_argument("--model", default=PROMPT_DEFAULT_MODEL)
    p.add_argument("--context-turns", type=_positive_int, default=20)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-output-chars", type=_positive_int, default=20000)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = make_argparser().parse_args(argv)
    shell = Hallunix(
        model=args.model,
        stream=not args.no_stream,
        context_turns=args.context_turns,
        temperature=args.temperature,
        max_output_chars=args.max_output_chars,
    )
    try:
        shell.run()
        return 0
    except Exception as e:
        print(f"hallunix: fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
