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
import json
import os
import readline  # noqa: F401  # still required for history on many systems
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Required deps; let ImportError surface if missing
from litellm import completion
# Rich line editing with custom key bindings
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings

NEOFETCH_TEMPLATE = r"""
       _          _           
      | |        (_)          
      | |__  _ __ _ _   _ _ __ 
      | '_ \| '__| | | | | '_ \ 
      | | | | |  | | |_| | | | |
      |_| |_|_|  |_|\__,_|_| |_|

OS: HallUnix x86_64"""


NEOFETCH_IF_SKIPPED = """
Host: HallServer  
Kernel: 5.15.0-58-generic  
Uptime: 2 days, 4 hours, 17 minutes  
Shell: zsh 5.8  
CPU: Intel Core i7-9700 @ 3.00GHz  
GPU: NVIDIA GeForce RTX 3060  
Memory: 16GB RAM  
Disk: 512GB SSD  
Resolution: 1920x1080
"""


NEOFETCH_GEN_SYSTEM_PROMPT = """
You generate a neofetch-style banner for a simulated Unix shell.
Return ONLY plain text (no Markdown, no code fences).
Requirements:
- DO NOT REPEAT the supplied ASCII art block or the initial fields
- Provide realistic neofetch fields: Host, Kernel, Uptime, Shell, CPU, GPU, Memory, etc.
- Keep tidy alignment; don't wrap excessively.
- Do NOT include any explanations or annotations.
- Do NOT align the fields, simply use 'Name: Value' with a single space.
"""

NEOFETCH_GEN_USER_PROMPT = """
User prompt:
{prompt_command}

Base ASCII art and initial fields:
{template}

Complete it now following the rules."""


DEFAULT_PROMPT_COMMAND = "user@hallunix:~$ "

SYSTEM_PROMPT = """
You are HallUnix, an AI-simulated Linux/Unix-like OS.
Act like a full OS shell: interpret each user line strictly as a command and
output ONLY the terminal output (stdout/stderr) for that command.
NEVER include explanations, Markdown, code fences, or surrounding quotes.
If a command would produce no output, return a blank string. If invalid,
return the typical Unix error. Assume a standard GNU userland.

State & environment: Start in /home/user with a plausible UNIX-like FS.
Maintain cwd and simulated file/process state consistently using prior
commands+outputs as truth. Do not claim '.' is missing.

Header & prompts: The session begins with the neofetch-style header below,
which already ends with a prompt and no trailing newline. Treat that exact
text as your first assistant message and continue from there. When emitting a
prompt (shell `$` prompt or Python `>>>`), end it with a single space and do
not add a trailing newline. Examples: 'user@hallunix:~$ ', '>>> '.

Large outputs: be realistic; truncate like a pager/head with ellipses when huge.
"""

# ---------- LLM-powered Bash completion ----------

# ---- Completion prompt templates (global) ----
COMPLETION_SYSTEM_PROMPT = """
You are a Bash command-completion engine for a Unix-like shell.
Respond with ONLY a strict JSON array of {n} strings — no explanations, no prose.
Each string must be a valid, complete Bash command line or short multi-line snippet.

Guidelines:
- Completions must begin EXACTLY with the user’s input.
- Provide completions with varying complexity.
  Start from trivial completions, but finish with complex and interesting ones.
- Favor realistic, non-trivial commands: pipelines (grep/awk/sed/jq), redirections,
  process substitution, find+xargs, tar, ssh/scp/rsync, git, docker, systemctl,
  journalctl or loops/conditionals 
- Multi-line snippets are allowed if natural (e.g., loops, here-docs).
- Suggestions should be useful and diverse.

You will be presented with whole converstaion history for context.
"""


COMPLETION_USER_PROMPT = """
Now it is time to generate completions!
You will act as the autocompletion engine.
Your suggestions should replace the ENTIRE current line (from beginning-of-line to cursor).

This is current line before cursor:
```
{line}
```
Return EXACTLY {n} diverse, high-quality suggestions as a JSON array of strings.
"""


class HallunixCompleter(Completer):
    def __init__(
        self,
        *,
        model: str,
        temperature: Optional[float],
        max_suggestions: int,
        hallunix: "Hallunix",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_suggestions = max(1, int(max_suggestions))
        self.hallunix = hallunix

    def _query_llm(self, line_prefix: str) -> list[str]:
        user_prompt = COMPLETION_USER_PROMPT.format(
            line=line_prefix,
            n=self.max_suggestions,
        )
        sys_prompt = COMPLETION_SYSTEM_PROMPT.format(n=self.max_suggestions)
        messages = self.hallunix.compose_messages_with_context(
            user_prompt = user_prompt,
            system_prompt = sys_prompt,
        )

        resp = completion(
            model=self.model,
            messages=messages,
            **(
                {"temperature": self.temperature}
                if self.temperature is not None
                else {}
            ),
        )
        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        cleaned = strip_code_fences(raw).strip()
        data = json.loads(cleaned)  # strict: no fallback

        if not isinstance(data, list):
            raise ValueError("Completion response is not a JSON array")
        out: list[str] = []
        for s in data[: self.max_suggestions]:
            if isinstance(s, str) and s.strip():
                # Preserve newlines for multi-line snippets; trim trailing spaces per line.
                out.append("\n".join(line.rstrip() for line in s.splitlines()))
        return out

    def get_completions(self, document, complete_event):
        line_before = document.current_line_before_cursor
        yield Completion("", display="thinking...")

        try:
            suggestions = self._query_llm(line_before)
        except Exception:
            yield Completion("", display="failed!")
            return  # LLM-only; yield nothing on error

        replace_from_bol = -len(line_before)
        for s in suggestions:
            yield Completion(
                s,
                start_position=replace_from_bol,
                display=s,
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
        completion_model: str,
        context_turns: int,
        system_prompt: str,
        temperature: Optional[float],
        max_output_chars: int,
        skip_neofetch: bool,
        prompt_command: str,
    ) -> None:
        self.model = model
        self.completion_model = completion_model

        self.skip_neofetch = skip_neofetch
        self.prompt_command = prompt_command

        self.context_turns = max(0, context_turns)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_chars = max_output_chars
        self.history: List[Turn] = []
        self._current_prompt: str = ""
        self._initial_header: str | None = None
        # prompt_toolkit session & key bindings
        self._kb = KeyBindings()

        @self._kb.add("c-j", eager=True)
        def _(event):
            # Insert literal newline without submitting
            event.current_buffer.insert_text("\n")

        @self._kb.add("enter", eager=True)
        def _(event):
            # Submit the current buffer
            event.current_buffer.validate_and_handle()

        self._session = PromptSession(key_bindings=self._kb, multiline=True)

        self._completer = HallunixCompleter(
            model=self.completion_model,
            temperature=self.temperature,
            max_suggestions=6,
            hallunix=self,
        )

    def _generate_neofetch_header(self) -> str:
        """
        Ask the LLM to complete the NEOFETCH base and return a header that
        ends with 'user@hallunix:~$ ' and NO trailing newline.
        Defensive fixes are applied if the model misses a constraint.
        """
        if self.skip_neofetch:
            return NEOFETCH_IF_SKIPPED.strip()

        messages = [
            {"role": "system", "content": NEOFETCH_GEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": NEOFETCH_GEN_USER_PROMPT.format(
                    prompt_command=self.prompt_command, template=NEOFETCH_TEMPLATE
                ),
            },
        ]
        kwargs = {"model": self.completion_model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        resp = completion(**kwargs)
        raw = resp["choices"][0]["message"]["content"] or ""
        txt = strip_code_fences(raw)
        return txt.strip()

    def run(self) -> None:
        print(NEOFETCH_TEMPLATE)
        header = self._generate_neofetch_header() + "\n"
        print(header)

        self._initial_header = "\n".join(
            [NEOFETCH_TEMPLATE, header, self.prompt_command]
        )
        self._current_prompt = self._normalize_prompt_for_input(self.prompt_command)

        while True:
            try:
                line = self._session.prompt(
                    self._current_prompt,
                    completer=self._completer,
                    complete_in_thread=True,
                    complete_while_typing=False,
                )
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

            if line.strip() in {"::exit"}:
                break
            if line.strip() == "::history":
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

    def compose_messages_with_context(self, user_prompt: str, system_prompt: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]

        assert self._initial_header is not None
        messages.append({"role": "assistant", "content": self._initial_header})

        if self.history and self.context_turns:
            recent = self.history[-self.context_turns :]
            for t in recent:
                messages.append({"role": "user", "content": t.command})
                messages.append({"role": "assistant", "content": t.output or ""})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _exec_with_llm(self, command: str) -> Tuple[str, bool]:
        messages = self.compose_messages_with_context(command, self.system_prompt)
        try:
            kwargs = {"model": self.model, "messages": messages}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            resp = completion(**kwargs)
            content = resp["choices"][0]["message"]["content"] or ""

            # Apply local truncation if necessary
            if len(content) >= self.max_output_chars:
                content = (
                    content[: self.max_output_chars]
                    + "\n[hallunix: output truncated]\n"
                )

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
    class SmartFormatter(argparse.RawDescriptionHelpFormatter):
        """"""

    p = argparse.ArgumentParser(
        prog="hallunix",
        description=(
            "Hallunix — an AI-simulated Unix-like environment over LiteLLM.\n"
            "Commands you type are sent to an LLM; only the terminal output is printed."
        ),
        formatter_class=SmartFormatter,
        epilog=textwrap.dedent(
            """
            Notes:
              • The header ends with the active prompt; type immediately.
              • Use ::exit to quit; ::history to print past commands.
              • Enter submits; Ctrl-J inserts a literal newline.

            Examples:
              hallunix --model gpt-4o-mini
              hallunix --model openrouter/anthropic/claude-3.5-sonnet
              hallunix --model gpt-4o-mini --completion-model gpt-4o-realtime-preview
              hallunix --context-turns 50 --max-output-chars 5000
            """
        ),
    )

    p.add_argument(
        "--model",
        required=True,
        help=(
            "Chat model identifier passed to LiteLLM for executing commands "
            "(e.g., 'gpt-4o-mini', 'openrouter/anthropic/claude-3.5-sonnet')."
        ),
    )
    p.add_argument(
        "--completion-model",
        default=None,
        help=(
            "Model used for interactive shell autocompletion. "
            "If not set, falls back to --model."
        ),
    )
    p.add_argument(
        "--context-turns",
        type=_positive_int,
        default=20,
        help=(
            "Number of most-recent turns to include as conversational context. "
            "Set to 0 to disable history."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature for both command execution and completion. "
            "Omit to use the provider default."
        ),
    )
    p.add_argument(
        "--max-output-chars",
        type=_positive_int,
        default=20000,
        help=(
            "Hard cap on characters printed from the model. When exceeded, "
            "remaining output is truncated with a notice."
        ),
    )
    p.add_argument(
        "--skip-neofetch",
        action=argparse.BooleanOptionalAction,
        help=(
            "Skip generating the LLM neofetch header; print a fixed fallback banner instead."
        ),
    )
    p.add_argument(
        "--prompt-command",
        type=str,
        default=DEFAULT_PROMPT_COMMAND,
        metavar="STRING",
        help=(
            "Exact prompt string shown before input and sent to the model when it "
            "emits a prompt. A single trailing space is added during input."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    print(">>> Hint: type '::exit' to exit")

    args = make_argparser().parse_args(argv)
    completion_model = args.completion_model or args.model

    shell = Hallunix(
        model=args.model,
        completion_model=completion_model,
        context_turns=args.context_turns,
        system_prompt=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_output_chars=args.max_output_chars,
        skip_neofetch=args.skip_neofetch,
        prompt_command=args.prompt_command,
    )
    try:
        shell.run()
        return 0
    except Exception as e:
        print(f"hallunix: fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
