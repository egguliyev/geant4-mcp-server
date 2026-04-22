# 🔬 RadDetect AI — Geant4 & Allpix^2 MCP Server

**AI-powered radiation detector simulation through natural language.**

A single-file [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that connects [Allpix^2](https://cern.ch/allpix-squared) and [Geant4](https://cern.ch/geant4) Monte Carlo simulation with Claude AI. Design, configure, run, and analyze pixelated radiation detector simulations by talking to Claude — no manual config writing needed.

Built by [Rad Detect AI](https://rad-ai.org) for the radiation detection and medical imaging community.

---

## What It Does

You talk to Claude. Claude calls the MCP tools. Allpix^2 runs your simulation.

```
You:    "Simulate a 2mm CdZnTe detector, 110µm pitch, 20µm gap, -800V, 60 keV gammas, 5000 events"

Claude: → calls geant4_generate_allpix2_config
        → writes main.conf + detector model + geometry
        → runs: allpix -c main.conf
        → returns ROOT output with 5M+ objects
```

---

## 9 Tools

| # | Tool | What it does |
|---|------|--------------|
| 1 | `geant4_generate_macro` | Generate Geant4 macro files (.mac) with geometry, physics, scoring |
| 2 | `geant4_generate_allpix2_config` | Create complete Allpix^2 configs for pixel detector simulation |
| 3 | `geant4_run_simulation` | Execute Allpix^2 or Geant4 simulations |
| 4 | `geant4_analyze_results` | Analyze output — energy spectra, spatial maps, charge sharing |
| 5 | `geant4_material_lookup` | Detector material database (CdZnTe, CdTe, Si, Ge, CsI, BGO, LYSO, GaAs) |
| 6 | `geant4_generate_xray_spectrum` | Polyenergetic X-ray spectrum generation (W, Mo, Rh anodes) |
| 7 | `geant4_list_files` | Browse workspace files |
| 8 | `geant4_physics_calculator` | Hecht equation, Fano resolution, charge sharing, attenuation |
| 9 | `geant4_workspace_status` | Check environment and available simulators |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Allpix^2](https://allpix-squared.docs.cern.ch/docs/02_installation/) (with Geant4)
- [Claude Code](https://code.claude.com) or [Claude Desktop](https://claude.ai/download)

### Install (3 commands)

```bash
mkdir -p ~/geant4-mcp-server && cd ~/geant4-mcp-server
python3 -m venv .venv && source .venv/bin/activate
pip install "mcp[cli]" pydantic httpx numpy uproot matplotlib scipy
```

Copy `geant4_mcp_server.py` into `~/geant4-mcp-server/`.

Verify:

```bash
python3 -c "from geant4_mcp_server import mcp; print('✅ Server OK:', mcp.name)"
```

### Connect to Claude Code

```bash
claude mcp add geant4-simulation -- ~/geant4-mcp-server/.venv/bin/python3 ~/geant4-mcp-server/geant4_mcp_server.py
```

Start Claude Code and type `/mcp` — you should see 9 tools from `geant4-simulation`.

### Connect to Claude Desktop

Create `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "geant4-simulation": {
      "command": "/home/YOUR_USER/geant4-mcp-server/.venv/bin/python3",
      "args": ["/home/YOUR_USER/geant4-mcp-server/geant4_mcp_server.py"],
      "env": {
        "GEANT4_MCP_WORKSPACE": "/home/YOUR_USER/geant4_mcp_workspace",
        "GEANT4_INSTALL_DIR": "/opt/geant4"
      }
    }
  }
}
```

Restart Claude Desktop.

---

## Example Conversations

**Simulate a CdZnTe photon-counting detector:**
> Generate an Allpix^2 config for a 2mm thick CdZnTe detector with 110µm pixel pitch and 20µm interpixel gap, biased at -800V. Simulate 60 keV gammas, 5000 events.

**Look up material properties:**
> Compare the µτ products and Fano factors of CdZnTe vs CdTe vs Silicon.

**Calculate detector performance:**
> What's the Fano-limited energy resolution of CdZnTe at 140 keV with 80 electrons of electronic noise?

**Generate an X-ray spectrum:**
> Generate a 120 kVp tungsten anode spectrum with 2.5mm Al filtration for CT simulation.

---

## Materials Database

Built-in properties for 9 detector materials:

| Material | Type | Z_eff | Density (g/cm³) |
|----------|------|-------|-----------------|
| CdZnTe | Semiconductor | 49.1 | 5.78 |
| CdTe | Semiconductor | 50.0 | 5.85 |
| Si | Semiconductor | 14 | 2.33 |
| Ge | Semiconductor | 32 | 5.32 |
| GaAs | Semiconductor | 32 | 5.32 |
| CsI:Tl | Scintillator | 54 | 4.51 |
| CsI:Na | Scintillator | 54 | 4.51 |
| BGO | Scintillator | 73 | 7.13 |
| LYSO | Scintillator | 65 | 7.10 |

Includes mobility, µτ products, Fano factors, bandgap, pair creation energy, light yield, and more.

---

## Physics Calculators

- **Hecht equation** — Charge collection efficiency vs bias voltage and thickness
- **Fano-limited energy resolution** — Fundamental resolution with electronic noise
- **Charge sharing estimation** — Diffusion-based interpixel charge sharing fraction
- **Attenuation length** — Photon mean free path in detector material

---

## Tested With

- Ubuntu 22.04 / 24.04 LTS
- Allpix^2 v3.0.0
- Geant4 11.2.2
- Python 3.11
- Claude Code v2.1

---

## Roadmap

- [x] Allpix^2 config generation + execution
- [x] Material database (9 materials)
- [x] Physics calculators (Hecht, Fano, charge sharing)
- [x] X-ray spectrum generator
- [ ] ROOT file analysis with uproot
- [ ] GATE integration for medical imaging
- [ ] GDML geometry import/export
- [ ] RadDetect AI Academy integration
- [ ] MTF / NPS / DQE calculation tools

---

## Author

**Elmaddin Guliyev, Ph.D.**
Founder, [Rad Detect AI](https://rad-ai.org)

Expertise: CdZnTe/CdTe photon-counting detectors, charge transport simulation, X-ray imaging systems, Monte Carlo (Geant4, Allpix^2).

- [LinkedIn — Rad Detect AI](https://www.linkedin.com/company/rad-detect-ai)
- [GitHub](https://github.com/egguliyev)

---

## License

MIT License
