# HallUnix

HallUnix is a fully hallucinated Unix-like OS environment.

HallUnix mimics a Linux shell/OS by routing all commands through an LLM via [LiteLLM](https://github.com/BerriAI/litellm).
It starts with a fresh OS session and responds to commands realistically — including interactive interpreters like
Python. Session history is ephemeral (per run) and no real commands are executed on your machine.

## Features

- **Mostly vibe-coded**: which means it might not work
- **LLM-powered OS**: every command is interpreted and answered by LLM
- **AI-powered autocompletion**: you don't have to type — use TAB to see AI suggestions
- **Authentic UX**: installation dialog and startup neofetch logo with system details
- **Interactive interpreters**: work well in Python (`python`) or similar REPLs naturally
- **Possible to exit**: by typing `::exit`

## How to use

Use with your own API keys and your favourite LLM:

```
export OPENAI_API_KEY="sk-..."
pip install -r requirements.txt
python hallunix.py
```

### Optional arguments:

#### `--config FILE`

Path to `.json` configuration file.
If your `.json` file contains all necessary fields, the installation prompt will be skipped.
Example of the content:

```json
{
  "environment-model": "gemini/gemini-2.5-pro",
  "environment-model-config": {
    "max_tokens": 4096
  },
  "assistant-model": "gemini/gemini-2.5-flash",
  "assistant-model-config": {
    "max_tokens": 4096
  },
  "os-ps1": "bob@ubuntu:~$",
  "os-details": {
    "OS": "Ubuntu x86_64",
    "CPU": "AMD Ryzen Threadripper 7980X 64-Cores",
    "GPU": "NVIDIA GeForce RTX 4090 [Discrete]"
  }
}
```

To see supported LLM input parameters, refer to [LiteLLM Documentation](https://docs.litellm.ai/docs/completion/input).

#### `--quick`

Skip generation of logo and OS details and use the default ones.

#### `--no-hints`

Skip printing hints at the start of the session.

## Examples

See examples below if you are too lazy (or poor) to try it out yourself.

### Installation
<img width="400" alt="image" src="https://github.com/user-attachments/assets/1477133c-94a6-435d-96e3-fcc718f1c30a" />

### Session
```
       _          _           
      | |        (_)          
      | |__  _ __ _ _   _ _ __ 
      | '_ \| '__| | | | | '_ \ 
      | | | | |  | | |_| | | | |
      |_| |_|_|  |_|\__,_|_| |_|

OS: HallUnix x86_64
Host: threadripper-7995WX  
Kernel: 5.15.0-60-generic  
Uptime: 2 days, 4 hours, 12 minutes  
Shell: bash 5.0.17  
CPU: AMD Ryzen Threadripper 7995WX 32-Core Processor  
GPU: NVIDIA GeForce RTX 3090  
Memory: 64 GiB  
Disk: 2 TB SSD  
Resolution: 3840x2160

bob@threadripper-7995WX:~$ ls
Documents
Downloads
Music
Pictures
Projects
Videos
bin
data.csv
notes.txt
script.sh
todo.md

bob@threadripper-7995WX:~$ cd Projects/work && python data_analysis.py < input/data.csv > output/results.txt

bob@threadripper-7995WX:~/Projects/work$ cat output/results.txt | head
Summary Report
==============
Rows: 10000
Columns: 6
Numeric columns: value,cost
Categorical columns: category,region
Date column: date
Missing values: value=42, cost=17
value mean=123.45, std=56.78, min=0.12, max=998.76
cost mean=45.67, std=12.34, min=1.23, max=123.45

bob@threadripper-7995WX:~/Projects/work$ ipython
Python 3.10.12 (main, Jun 12 2024, 14:00:00) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
IPython 8.20.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from data import data_utils

In [2]: df = data_utils.load_dataset_from_hf()

In [3]: df.head()
Out[3]: 
   id        date     category region   value   cost
0   1 2023-01-01  Electronics  North  123.45  45.67
1   2 2023-01-02      Grocery   West   98.76  23.45
2   3 2023-01-03     Clothing  South    0.12  12.34
3   4 2023-01-04  Electronics   East  250.00  56.78
4   5 2023-01-05      Grocery  North  175.90  34.56

In [4]: exit

bob@threadripper-7995WX:~/Projects/work$ ::exit
```
