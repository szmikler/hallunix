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
import readline  # noqa: F401  # still required for history on many systems
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm import completion
from prompt_toolkit import PromptSession, application, key_binding, layout, widgets
from prompt_toolkit.completion import Completer, Completion


def multiline_to_dict(config: str) -> dict[str, str]:
    cfg = {}
    lines = config.strip().split("\n")
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
        else:
            k = line
            v = " "
        try:
            cfg[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError:
            cfg[k.strip()] = v.strip()
    return cfg


def dict_to_multiline(cfg: dict[str, str] | None) -> str | None:
    if cfg is None:
        return None
    lines = [f"{k}: {v}" for k, v in cfg.items()]
    return "\n".join(lines)


class HallunixConfig:
    quick: bool = False

    env_model: str | None = None
    env_model_config: dict | None = None

    ast_model: str | None = None
    ast_model_config: dict | None = None

    os_ps1: str | None = None
    os_details: dict | None = None

    def from_dict(self, config: dict):
        self.env_model = config.get("environment-model", self.env_model)
        self.env_model_config = config.get(
            "environment-model-config", self.env_model_config
        )
        self.ast_model = config.get("assistant-model", self.ast_model)
        self.ast_model_config = config.get(
            "assistant-model-config", self.ast_model_config
        )
        self.os_ps1 = config.get("os-ps1", self.os_ps1)
        self.os_details = config.get("os-details", self.os_details)

        if isinstance(self.env_model_config, str):
            self.env_model_config = multiline_to_dict(self.env_model_config)
        if isinstance(self.ast_model_config, str):
            self.ast_model_config = multiline_to_dict(self.ast_model_config)
        if isinstance(self.os_details, str):
            self.os_details = multiline_to_dict(self.os_details)

    def all_set(self):
        return (
            isinstance(self.env_model, str)
            and isinstance(self.ast_model, str)
            and isinstance(self.env_model_config, dict)
            and isinstance(self.ast_model_config, dict)
            and isinstance(self.os_ps1, str)
            and isinstance(self.os_details, dict)
        )

    def __repr__(self):
        return (
            "HallunixConfig("
            f"env_model={self.env_model!r}, "
            f"env_model_kwds={self.env_model_config!r}, "
            f"ast_model={self.ast_model!r}, "
            f"ast_model_kwds={self.ast_model_config!r}, "
            f"os_ps1={self.os_ps1!r}, "
            f"os_details={self.os_details!r}"
            ")"
        )


CONFIG = HallunixConfig()


QUICK_INIT = r"""
  _   _       _ _   _       
 | | | | __ _| | |_(_)_ __  
 | |_| |/ _` | | __| | '_ \ 
 |  _  | (_| | | |_| | | | |
 |_| |_|\__,_|_|\__|_|_| |_|
   _   _       _           
  | | | |_ __ | | ___  ___ 
  | | | | '_ \| |/ _ \/ __|
  | |_| | | | | |  __/\__ \
   \___/|_| |_|_|\___||___/

OS: HallUnix x86_64
Host: HallMachine
Kernel: 5.18.3-hallunix
Uptime: 3 days, 4 hours, 12 mins
Packages: 1578
Shell: bash 5.2.15
Resolution: 1920x1080
DE: GlowWM 1.4
WM: GlowWM
CPU: AMD Ryzen 7 5800X (8) @ 3.8GHz
GPU: Radeon RX 6800
Memory: 12437MiB / 32768MiB

user@hallunix:~$
"""


NEOFETCH_ART_SYSTEM_PROMPT = r"""
You are an Neofetch ASCII art generator for a simulated Unix shell. 
Output requirements:
- Output ONLY ASCII art in plain text (no Markdown, no code fences, no explanations).
- The art must depict something thematically related to Unix, Linux, or computing.
- The art must be at most 10 lines tall.
- The art must be at most 50 characters wide.
- The art must be properly aligned, consistent, and visually appealing.
- Use only standard ASCII characters (no Unicode, no extended symbols).
- Do not include labels, comments, or extra text.
- Always return a complete ASCII art block, never partial or broken art.
- If you see a real OS, use its real Neofetch ASCII art.
- If you see a non-real OS, create a plausible Neofetch ASCII art for it.

Examples of valid ASCII art:
   _          _           
  | |        (_)          
  | |__  _ __ _ _   _ _ __ 
  | '_ \| '__| | | | | '_ \ 
  | | | | |  | | |_| | | | |
  |_| |_|_|  |_|\__,_|_| |_|
   ____        _       
  |  _ \ _   _| | ___  
  | |_) | | | | |/ _ \ 
  |  __/| |_| | |  __/ 
  |_|    \__,_|_|\___|  
   .--.     
  |o_o |    
  |:_/ |    
 //   \ \   
(|     | )  
/'\_   _/`\ 
\___)=(___/ 
"""

NEOFETCH_ART_USER_PROMPT = """
Provided initial data:
{initial_data_str}

Generate an ASCII logo or symbol inspired by the data (if relevant). Ignore non-relevant data.
"""

NEOFETCH_DATA_SYSTEM_PROMPT = """
You generate a neofetch-style system info for a simulated Unix shell.
Return ONLY plain text (no Markdown, no code fences).
Requirements:
- You must include ALL the provided initial data
- Make sure you provide realistic neofetch fields like: Host, Kernel, Uptime, Shell, CPU, GPU, Memory, etc.
- Provide fields in the correct order, like they would appear in a real terminal
- Every field must be placed in a separate line
- Do NOT include any explanations or annotations
- Do NOT align the fields with multiple spaces. Instead use 'Name: Value' with a single space
- Do NOT include anything except for the fields, for example do not include line with 'username@hostname'
"""

NEOFETCH_DATA_USER_PROMPT = """
Provided initial data:
{initial_data_str}

Complete missing fields following the rules.
"""

ENVIRONMENT_SYSTEM_PROMPT = """
You are an AI-simulated Linux/Unix-like OS.
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
text as your first assistant message and continue from there. Usually after
command output there should be a newline and a prompt. When emitting the prompt
(shell `$` prompt or Python `>>>`), end it with a single space and do not add
a trailing newline. Examples: 'user@hallunix:~$ ', '>>> '.

Large outputs: be realistic; truncate like a pager/head with ellipses when huge.
"""

# ---------- LLM-powered Bash completion ----------

# ---- Completion prompt templates (global) ----
COMPLETION_SYSTEM_PROMPT = """
You are a command completion engine for a novel OS.
Respond with ONLY a strict JSON array of {n} strings — no explanations, no prose.
Each string must be a valid, complete Bash command line or short multi-line snippet.
You will be presented with the entire conversation history for context.

Guidelines:
- Provide many different completions to the user.
- All completions must begin EXACTLY with the user’s input.
- Provide completions with varying complexity.
  Start from trivial completions, but finish with complex and interesting ones.
- Favor realistic, non-trivial commands: pipelines (grep/awk/sed/jq), redirections,
  process substitution, find+xargs, tar, ssh/scp/rsync, git, docker, systemctl,
  journalctl or loops/conditionals 
- Multi-line snippets are allowed if natural (e.g., loops, here-docs).
- Suggestions should be useful and diverse.
- Suggestions should match the current context.
  If the user is currently in Python interpreter, it should suggest Python commands.
  Same goes for other interactive environments.
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


def generate_neofetch_data() -> str:
    """
    Ask the LLM to complete the NEOFETCH base and return a header that ends with a prompt.
    """
    initial_data_str = dict_to_multiline(CONFIG.os_details)
    user_prompt_str = NEOFETCH_DATA_USER_PROMPT.format(
        initial_data_str=initial_data_str
    )

    messages = [
        {"role": "system", "content": NEOFETCH_DATA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_str},
    ]
    resp = completion(
        model=CONFIG.ast_model, messages=messages, **CONFIG.ast_model_config
    )
    raw = resp["choices"][0]["message"]["content"] or ""
    return strip_code_fences(raw.strip()) + "\n"


def generate_neofetch_art() -> str:
    """
    Ask the LLM to generate a simple ASCII art for the neofetch header.
    """
    initial_data_str = dict_to_multiline(CONFIG.os_details)
    user_prompt_str = NEOFETCH_ART_USER_PROMPT.format(initial_data_str=initial_data_str)

    messages = [
        {"role": "system", "content": NEOFETCH_ART_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_str},
    ]
    resp = completion(
        model=CONFIG.ast_model, messages=messages, **CONFIG.ast_model_config
    )
    raw = resp["choices"][0]["message"]["content"] or ""
    return strip_code_fences(raw.rstrip()) + "\n"


class HallunixCompleter(Completer):
    def __init__(
        self,
        *,
        hallunix: "HallunixSession",
        max_suggestions: int,
    ) -> None:
        self.hallunix = hallunix
        self.max_suggestions = max(1, int(max_suggestions))

    def _query_llm(self, line_prefix: str) -> list[str]:
        user_prompt = COMPLETION_USER_PROMPT.format(
            line=line_prefix,
            n=self.max_suggestions,
        )
        sys_prompt = COMPLETION_SYSTEM_PROMPT.format(n=self.max_suggestions)
        messages = self.hallunix.compose_messages_with_context(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
        )

        resp = completion(
            model=CONFIG.ast_model,
            messages=messages,
            **CONFIG.ast_model_config,
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


@dataclass
class Turn:
    command: str
    output: str


class HallunixSession:
    def __init__(
        self,
        max_context_turns: int,
    ) -> None:

        self.max_context_turns = max(0, max_context_turns)
        self.history: List[Turn] = []
        self._current_prompt: str = ""
        self._session_header: str | None = None
        self._kb = key_binding.KeyBindings()

        @self._kb.add("c-j", eager=True)
        def _(event):  # Insert literal newline without submitting
            event.current_buffer.insert_text("\n")

        @self._kb.add("enter", eager=True)
        def _(event):  # Submit the current buffer
            event.current_buffer.validate_and_handle()

        self._session = PromptSession(key_bindings=self._kb, multiline=True)
        self._completer = HallunixCompleter(hallunix=self, max_suggestions=6)

    def run(self) -> None:
        if CONFIG.quick:
            screen, prompt = split_screen_and_prompt(QUICK_INIT.strip("\n"))
            self._current_prompt = self._normalize_prompt_for_input(prompt)
            self._session_header = QUICK_INIT
            print(screen, end="")
        else:
            neofetch_art = generate_neofetch_art()
            print(neofetch_art)

            neofetch_data = generate_neofetch_data()
            print(neofetch_data)

            self._current_prompt = self._normalize_prompt_for_input(CONFIG.os_ps1)
            self._session_header = "\n".join(
                (neofetch_art, neofetch_data, self._current_prompt)
            )

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

            if line.strip() == "::exit":
                break
            if line.strip() == "::history":
                self._print_history()
                continue

            output = self._environment_action(line)
            if output is not None:
                screen, prompt = split_screen_and_prompt(output)
                if screen:
                    print(screen, end="")
                if prompt:
                    self._current_prompt = self._normalize_prompt_for_input(prompt)
            self.history.append(Turn(command=line, output=output or ""))

    def compose_messages_with_context(
        self, user_prompt: str, system_prompt: str
    ) -> list:
        messages = [{"role": "system", "content": system_prompt}]

        assert self._session_header is not None
        messages.append({"role": "assistant", "content": self._session_header})

        if self.history and self.max_context_turns:
            recent = self.history[-self.max_context_turns :]
            for t in recent:
                messages.append({"role": "user", "content": t.command})
                messages.append({"role": "assistant", "content": t.output or ""})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _environment_action(self, command: str) -> str | None:
        messages = self.compose_messages_with_context(
            command or " ", ENVIRONMENT_SYSTEM_PROMPT
        )
        assert len(messages) > 0 and messages[-1]["role"] != "assistant"

        try:
            resp = completion(
                model=CONFIG.env_model, messages=messages, **CONFIG.env_model_config
            )
            content = resp["choices"][0]["message"]["content"] or ""
            cleaned = strip_code_fences(content)
            return cleaned

        except (KeyboardInterrupt, IndexError, KeyError):
            print()
            return None

    def _normalize_prompt_for_input(self, prompt: str) -> str:
        return prompt.rstrip() + " " if prompt else ""

    def _print_history(self) -> None:
        header = "idx\toutput\tinput"
        print(header)
        print("---")
        for i, t in enumerate(self.history, start=1):
            print(f"{i}\t{len(t.output)}\t{t.command}")


def get_installation_app():
    text_areas = {}

    def labeled_text_area(display, label, **kwds):
        text_areas[label] = widgets.TextArea(**kwds, focus_on_click=True)
        return layout.HSplit([widgets.Label(text=display), text_areas[label]])

    fields = [
        labeled_text_area(
            label="environment-model",
            display="Environment Model",
            text=CONFIG.env_model or "openai/gpt-5",
            height=2,
        ),
        labeled_text_area(
            label="environment-model-config",
            display="Environment Model Configuration",
            text=dict_to_multiline(CONFIG.env_model_config) or "max_tokens: 4096",
            height=6,
        ),
        labeled_text_area(
            label="assistant-model",
            display="Assistant Model",
            text=CONFIG.ast_model or "openai/gpt-4.1-mini",
            height=2,
        ),
        labeled_text_area(
            label="assistant-model-config",
            display="Assistant Model Configuration",
            text=dict_to_multiline(CONFIG.ast_model_config) or "max_tokens: 4096",
            height=6,
        ),
        labeled_text_area(
            label="os-ps1",
            display="OS PS1 Prompt",
            text=CONFIG.os_ps1 or "user@hallunix:~$",
            height=2,
        ),
        labeled_text_area(
            label="os-details",
            display="OS Details",
            text=dict_to_multiline(CONFIG.os_details) or "OS: HallUnix x86_64",
            height=6,
        ),
    ]

    dialog = widgets.Dialog(
        title="HallUnix Installation",
        body=layout.HSplit(
            [
                widgets.Button(
                    text="Continue", handler=lambda: app.exit(result=get_result())
                ),
                *fields,
            ],
            padding=1,
            width=60,
        ),
    )

    def get_result():
        return {label: text_area.text for label, text_area in text_areas.items()}

    root = layout.FloatContainer(
        content=layout.Window(),
        floats=[layout.Float(content=dialog)],
    )

    keys = key_binding.KeyBindings()

    @keys.add("c-c")
    @keys.add("c-q")
    def _(event):
        sys.exit(1)

    app = application.Application(
        layout=layout.Layout(root),
        mouse_support=True,
        full_screen=True,
        key_bindings=keys,
    )
    return app


HINTS_STR = """
type ::exit to exit
type ::history to display command history
press <ctrl+j> to insert a newline
press <tab> to trigger auto-completion
"""


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="hallunix",
        description=(
            "Hallunix — hallucinated Unix-like environment.\n"
            "An AI-simulated shell that interprets commands through an LLM, "
            "rendering realistic terminal output with a neofetch-style header."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        metavar="FILE",
        help=(
            "Path to JSON configuration file specifying models, prompts, "
            "and OS details. If omitted, an interactive installer is launched."
        ),
    )

    parser.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a built-in static neofetch header instead of generating one via models. "
            "Still requires environment/assistant models to run. Default: off."
        ),
    )

    parser.add_argument(
        "--hints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show usage hints on startup (controls, ::exit, ::history). Default: on.",
    )

    args = parser.parse_args(argv)
    CONFIG.quick = args.quick

    if args.hints:
        tutorial = textwrap.indent(HINTS_STR, prefix=">>> ").strip()
        print(tutorial)
        print()

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        CONFIG.from_dict(config)
    if not CONFIG.all_set():
        app = get_installation_app()
        config = app.run()
        CONFIG.from_dict(config)
    assert CONFIG.all_set(), str(CONFIG)

    shell = HallunixSession(max_context_turns=100)
    shell.run()


if __name__ == "__main__":
    raise SystemExit(main())
