# Autonomous Warehouse Logistics

A multi-agent warehouse simulation built with Python and MuJoCo.

## Requirements

- Python 3.9 or newer
- A display is required for the interactive viewer

## Install

```bash
git clone https://github.com/neta-shami/Autonomous-Warehouse-Logistics-Project.git
cd Autonomous-Warehouse-Logistics-Project

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Headless simulation:

```bash
.venv/bin/python main.py
```

This runs the scripted simulation for 1,000 logic ticks and prints a metrics
report.

Interactive 3D viewer:

```bash
# macOS
.venv/bin/mjpython visualize.py

# Linux or Windows
.venv/bin/python visualize.py
```

On macOS, use `mjpython` because MuJoCo requires its viewer to run on the
main thread.

## Interactive commands

Enter these commands at the `sim>` prompt:

| Command | Effect |
| --- | --- |
| `STORE pkg_id x,y tier` | Spawn a package at the inbound dock and store it on a shelf |
| `RETRIEVE pkg_id x,y tier` | Retrieve a package from a shelf and deliver it to the outbound dock |
| `MOVE src_x,src_y src_tier tgt_x,tgt_y tgt_tier` | Move a package between shelf slots |
| `status` | Show robot states, positions, payloads, and throughput |
| `help` | Show the command list |

Example:

```text
STORE p8 4,4 0
```
