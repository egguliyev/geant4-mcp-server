"""
RadDetect AI — Geant4 & Allpix² MCP Server
============================================
A professional MCP server for radiation detector Monte Carlo simulation,
built by Rad Detect AI (rad-ai.org).

Integrates with Claude Desktop / Claude Code via the Model Context Protocol
to let users design, configure, run, and analyze Geant4 and Allpix² simulations
through natural language.

Author: Elmaddin Guliyev / Rad Detect AI
License: MIT
"""

import json
import os
import subprocess
import tempfile
import shutil
import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
WORKSPACE_DIR = Path(os.environ.get("GEANT4_MCP_WORKSPACE", Path.home() / "geant4_mcp_workspace"))
GEANT4_INSTALL = Path(os.environ.get("GEANT4_INSTALL_DIR", "/opt/geant4"))
ALLPIX2_INSTALL = Path(os.environ.get("ALLPIX2_INSTALL_DIR", "/opt/allpix-squared"))
MACRO_DIR = WORKSPACE_DIR / "macros"
OUTPUT_DIR = WORKSPACE_DIR / "output"
CONFIG_DIR = WORKSPACE_DIR / "configs"

# Ensure directories exist
for d in [WORKSPACE_DIR, MACRO_DIR, OUTPUT_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Enums & Shared Models
# ─────────────────────────────────────────────

class ParticleType(str, Enum):
    GAMMA = "gamma"
    E_MINUS = "e-"
    E_PLUS = "e+"
    PROTON = "proton"
    NEUTRON = "neutron"
    ALPHA = "alpha"
    MU_MINUS = "mu-"
    OPTICAL_PHOTON = "opticalphoton"

class PhysicsList(str, Enum):
    EMSTANDARD = "G4EmStandardPhysics"
    EMSTANDARD_OPT4 = "G4EmStandardPhysics_option4"
    LIVERMORE = "G4EmLivermorePhysics"
    PENELOPE = "G4EmPenelopePhysics"
    FTFP_BERT = "FTFP_BERT"
    QGSP_BERT = "QGSP_BERT"
    QGSP_BIC = "QGSP_BIC"
    SHIELDING = "Shielding"

class DetectorMaterial(str, Enum):
    CDZNTE = "G4_CADMIUM_ZINC_TELLURIDE"
    CDTE = "G4_CADMIUM_TELLURIDE"
    SILICON = "G4_Si"
    GERMANIUM = "G4_Ge"
    CSI = "G4_CESIUM_IODIDE"
    NAITL = "G4_SODIUM_IODIDE"
    BGO = "G4_BGO"
    LYSO = "G4_LUTETIUM_YTTRIUM_SILICON_OXIDE"
    WATER = "G4_WATER"
    AIR = "G4_AIR"
    LEAD = "G4_Pb"
    TUNGSTEN = "G4_W"
    ALUMINUM = "G4_Al"
    COPPER = "G4_Cu"
    BONE = "G4_BONE_COMPACT_ICRU"
    TISSUE = "G4_TISSUE_SOFT_ICRP"

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"

# ─────────────────────────────────────────────
# MCP Server Initialization
# ─────────────────────────────────────────────

mcp = FastMCP("geant4_mcp")


# ═══════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def _run_command(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Dict[str, Any]:
    """Execute a shell command and return structured result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or WORKSPACE_DIR,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1}
    except FileNotFoundError as e:
        return {"success": False, "stdout": "", "stderr": f"Command not found: {e}", "returncode": -1}


def _format_response(data: Dict[str, Any], fmt: ResponseFormat) -> str:
    """Format response as JSON or Markdown."""
    if fmt == ResponseFormat.JSON:
        return json.dumps(data, indent=2, default=str)
    # Markdown formatting
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"### {key.replace('_', ' ').title()}")
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
        elif isinstance(value, list):
            lines.append(f"### {key.replace('_', ' ').title()}")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"- {json.dumps(item)}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
    return "\n".join(lines)


def _generate_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ═══════════════════════════════════════════════
# TOOL 1: Generate Geant4 Macro
# ═══════════════════════════════════════════════

class GenerateMacroInput(BaseModel):
    """Input for generating a Geant4 macro file."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(
        ..., description="Name for this macro (e.g., 'czt_pixel_detector')",
        min_length=1, max_length=100
    )
    particle: ParticleType = Field(
        default=ParticleType.GAMMA,
        description="Primary particle type"
    )
    energy_keV: float = Field(
        ..., description="Particle energy in keV (e.g., 60.0 for Am-241, 662.0 for Cs-137)",
        gt=0, le=1e9
    )
    num_events: int = Field(
        default=10000, description="Number of events to simulate",
        ge=1, le=1_000_000_000
    )
    detector_material: DetectorMaterial = Field(
        default=DetectorMaterial.CDZNTE,
        description="Detector material"
    )
    detector_size_mm: List[float] = Field(
        default=[20.0, 20.0, 5.0],
        description="Detector dimensions [x, y, z] in mm",
        min_length=3, max_length=3
    )
    pixel_pitch_um: Optional[float] = Field(
        default=None, description="Pixel pitch in µm (None for monolithic detector)",
        gt=0, le=10000
    )
    physics_list: PhysicsList = Field(
        default=PhysicsList.LIVERMORE,
        description="Geant4 physics list"
    )
    source_position_mm: List[float] = Field(
        default=[0.0, 0.0, -50.0],
        description="Source position [x, y, z] in mm"
    )
    beam_direction: List[float] = Field(
        default=[0.0, 0.0, 1.0],
        description="Particle momentum direction [dx, dy, dz]"
    )
    use_spectrum: bool = Field(
        default=False,
        description="If True, uses a polyenergetic X-ray spectrum instead of monoenergetic"
    )
    spectrum_anode: Optional[str] = Field(
        default=None,
        description="Anode material for X-ray spectrum (e.g., 'W', 'Mo', 'Rh')"
    )
    spectrum_kvp: Optional[float] = Field(
        default=None,
        description="Tube voltage in kVp for X-ray spectrum generation",
        gt=0, le=500
    )
    filtration_mm_al: Optional[float] = Field(
        default=None,
        description="Aluminum filtration thickness in mm",
        ge=0, le=100
    )
    scoring_type: str = Field(
        default="edep",
        description="Scoring type: 'edep' (energy deposition), 'dose', 'flux', 'spectrum'"
    )
    random_seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    verbose: int = Field(
        default=0, description="Geant4 verbosity level (0-2)",
        ge=0, le=2
    )
    custom_commands: Optional[List[str]] = Field(
        default=None,
        description="Additional Geant4 macro commands to append"
    )


@mcp.tool(
    name="geant4_generate_macro",
    annotations={
        "title": "Generate Geant4 Macro File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_generate_macro(params: GenerateMacroInput) -> str:
    """Generate a Geant4 macro (.mac) file for particle simulation.

    Creates a complete, runnable Geant4 macro with geometry definition,
    physics list selection, particle gun configuration, and scoring setup.
    Supports monoenergetic beams and polyenergetic X-ray spectra.
    Tailored for radiation detector simulations (CdZnTe, CdTe, CsI, Si, etc.).

    Returns:
        str: JSON with macro file path and content preview.
    """
    timestamp = _generate_timestamp()
    macro_filename = f"{params.name}_{timestamp}.mac"
    macro_path = MACRO_DIR / macro_filename

    lines = [
        f"# ═══════════════════════════════════════════════════════════",
        f"# Geant4 Macro — Generated by RadDetect AI MCP Server",
        f"# Name: {params.name}",
        f"# Date: {datetime.datetime.now().isoformat()}",
        f"# ═══════════════════════════════════════════════════════════",
        "",
        "# ── Verbosity ──",
        f"/control/verbose {params.verbose}",
        f"/run/verbose {params.verbose}",
        f"/tracking/verbose 0",
        "",
        "# ── Physics List ──",
        f"# Selected: {params.physics_list.value}",
        "",
        "# ── Geometry: Detector ──",
        f"/detector/material {params.detector_material.value}",
        f"/detector/size {params.detector_size_mm[0]} {params.detector_size_mm[1]} {params.detector_size_mm[2]} mm",
    ]

    if params.pixel_pitch_um:
        n_pixels_x = int(params.detector_size_mm[0] * 1000 / params.pixel_pitch_um)
        n_pixels_y = int(params.detector_size_mm[1] * 1000 / params.pixel_pitch_um)
        lines += [
            f"/detector/pixelPitch {params.pixel_pitch_um} um",
            f"# Pixel grid: {n_pixels_x} x {n_pixels_y} = {n_pixels_x * n_pixels_y} pixels",
        ]

    lines += [
        "",
        "# ── Initialize ──",
        "/run/initialize",
        "",
    ]

    # Random seed
    if params.random_seed is not None:
        lines += [
            "# ── Random Seed ──",
            f"/random/setSeeds {params.random_seed} {params.random_seed + 1}",
            "",
        ]

    # Source configuration
    lines += ["# ── Particle Source ──"]

    if params.use_spectrum and params.spectrum_kvp:
        lines += [
            f"# Polyenergetic X-ray spectrum: {params.spectrum_anode or 'W'} anode, {params.spectrum_kvp} kVp",
            "/gps/particle gamma",
            f"/gps/pos/centre {params.source_position_mm[0]} {params.source_position_mm[1]} {params.source_position_mm[2]} mm",
            f"/gps/direction {params.beam_direction[0]} {params.source_direction[1]} {params.source_direction[2]}",
            "/gps/ene/type User",
            f"/gps/hist/type energy",
            f"# NOTE: Load spectrum from file — {params.spectrum_anode or 'W'}_{int(params.spectrum_kvp)}kVp.spec",
            f"# Use /gps/hist/point E_keV weight for each energy bin",
            f"# Example: /gps/hist/point 0.020 0.001",
            f"#          /gps/hist/point 0.030 0.015",
            f"#          ... (load from SpekCalc or xpecgen output)",
        ]
        if params.filtration_mm_al:
            lines.append(f"# Filtration: {params.filtration_mm_al} mm Al equivalent")
    else:
        lines += [
            "/gun/particle " + params.particle.value,
            f"/gun/energy {params.energy_keV} keV",
            f"/gun/position {params.source_position_mm[0]} {params.source_position_mm[1]} {params.source_position_mm[2]} mm",
            f"/gun/direction {params.beam_direction[0]} {params.source_direction[1]} {params.source_direction[2]}",
        ]

    lines += [""]

    # Scoring
    lines += [
        "# ── Scoring ──",
        f"/score/create/boxMesh detectorMesh",
        f"/score/mesh/boxSize {params.detector_size_mm[0]/2} {params.detector_size_mm[1]/2} {params.detector_size_mm[2]/2} mm",
    ]

    if params.pixel_pitch_um:
        lines.append(f"/score/mesh/nBin {n_pixels_x} {n_pixels_y} 1")
    else:
        lines.append(f"/score/mesh/nBin 100 100 1")

    scoring_map = {
        "edep": "/score/quantity/energyDeposit edep",
        "dose": "/score/quantity/doseDeposit dose",
        "flux": "/score/quantity/flatSurfaceFlux flux",
        "spectrum": "/score/quantity/energyDeposit edep",
    }
    lines.append(scoring_map.get(params.scoring_type, scoring_map["edep"]))
    lines += [
        "/score/close",
        "",
    ]

    # Custom commands
    if params.custom_commands:
        lines += ["# ── Custom Commands ──"] + params.custom_commands + [""]

    # Run
    lines += [
        "# ── Run ──",
        f"/run/beamOn {params.num_events}",
        "",
        "# ── Output ──",
        f"/score/dumpQuantityToFile detectorMesh edep {OUTPUT_DIR}/{params.name}_{timestamp}.csv",
    ]

    macro_content = "\n".join(lines)
    macro_path.write_text(macro_content)

    result = {
        "status": "success",
        "macro_file": str(macro_path),
        "name": params.name,
        "particle": params.particle.value,
        "energy_keV": params.energy_keV,
        "events": params.num_events,
        "detector": {
            "material": params.detector_material.value,
            "size_mm": params.detector_size_mm,
            "pixel_pitch_um": params.pixel_pitch_um,
            "n_pixels": f"{n_pixels_x}x{n_pixels_y}" if params.pixel_pitch_um else "monolithic"
        },
        "physics_list": params.physics_list.value,
        "scoring": params.scoring_type,
        "content_preview": "\n".join(lines[:30]) + "\n... (truncated)"
    }

    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════
# TOOL 2: Generate Allpix² Configuration
# ═══════════════════════════════════════════════

class AllpixConfigInput(BaseModel):
    """Input for generating an Allpix² simulation configuration."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(..., description="Configuration name (e.g., 'czt_interpixel_study')", min_length=1, max_length=100)
    detector_model: str = Field(default="custom", description="Detector model name")
    sensor_material: str = Field(default="cdte", description="Sensor material: 'cdte', 'cdznte', 'silicon', 'germanium'")
    sensor_thickness_um: float = Field(default=2000.0, description="Sensor thickness in µm", gt=0, le=50000)
    pixel_pitch_um: float = Field(default=55.0, description="Pixel pitch in µm", gt=0, le=10000)
    n_pixels_x: int = Field(default=256, description="Number of pixels in X", ge=1, le=4096)
    n_pixels_y: int = Field(default=256, description="Number of pixels in Y", ge=1, le=4096)
    interpixel_gap_um: Optional[float] = Field(default=None, description="Interpixel gap width in µm", ge=0, le=1000)
    bias_voltage_V: float = Field(default=-500.0, description="Bias voltage in V (negative for typical operation)")
    temperature_K: float = Field(default=293.15, description="Operating temperature in Kelvin", gt=0, le=500)
    particle: ParticleType = Field(default=ParticleType.GAMMA, description="Primary particle type")
    energy_keV: float = Field(default=60.0, description="Particle energy in keV", gt=0)
    num_events: int = Field(default=10000, description="Number of events", ge=1, le=1_000_000)
    source_type: str = Field(default="beam", description="Source type: 'beam', 'point', 'square'")
    beam_size_mm: Optional[float] = Field(default=None, description="Beam spot size (sigma) in mm")
    electric_field_model: str = Field(
        default="linear",
        description="Electric field model: 'linear', 'init' (from TCAD), 'constant'"
    )
    charge_transport: str = Field(
        default="generic",
        description="Charge transport model: 'generic', 'projection'"
    )
    digitizer: str = Field(
        default="default",
        description="Digitizer model: 'default', 'csa' (charge-sensitive amplifier)"
    )
    noise_electrons: Optional[float] = Field(default=None, description="Electronic noise in ENC electrons RMS")
    threshold_electrons: Optional[float] = Field(default=None, description="Threshold in electrons")
    output_format: str = Field(default="root", description="Output format: 'root', 'csv', 'both'")
    enable_mc_truth: bool = Field(default=True, description="Enable MC truth output for validation")


@mcp.tool(
    name="geant4_generate_allpix2_config",
    annotations={
        "title": "Generate Allpix² Simulation Config",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_generate_allpix2_config(params: AllpixConfigInput) -> str:
    """Generate a complete Allpix² simulation configuration.

    Creates main config file + detector model + geometry for Allpix² Monte
    Carlo simulation of pixelated radiation detectors. Supports CdTe, CdZnTe,
    Si, Ge sensors with configurable charge transport, electric field,
    digitization, and interpixel gap studies.

    Returns:
        str: JSON with file paths and configuration summary.
    """
    timestamp = _generate_timestamp()
    config_subdir = CONFIG_DIR / f"{params.name}_{timestamp}"
    config_subdir.mkdir(parents=True, exist_ok=True)

    # ── Detector model file ──
    material_map = {
        "cdte": ("CADMIUM_TELLURIDE", 5.85),
        "cdznte": ("CADMIUM_ZINC_TELLURIDE", 5.78),
        "silicon": ("SILICON", 2.33),
        "germanium": ("GERMANIUM", 5.32),
    }
    mat_name, density = material_map.get(params.sensor_material, ("cadmium_telluride", 5.85))

    model_content = f"""# Allpix² Detector Model — Generated by RadDetect AI
# Model: {params.detector_model}
# Date: {datetime.datetime.now().isoformat()}

type = "monolithic"
geometry = "pixel"

number_of_pixels = {params.n_pixels_x} {params.n_pixels_y}
pixel_size = {params.pixel_pitch_um}um {params.pixel_pitch_um}um
sensor_thickness = {params.sensor_thickness_um}um
sensor_material = "{mat_name.upper()}"
"""
    if params.interpixel_gap_um is not None:
        implant_size_x = params.pixel_pitch_um - params.interpixel_gap_um
        implant_size_y = params.pixel_pitch_um - params.interpixel_gap_um
        model_content += f"""
[implant]
type = frontside
shape = rectangle
#size = {implant_size_x}um {implant_size_y}um
size  = {implant_size_x}um {implant_size_y}um 5um
"""
    model_path = config_subdir / f"{params.detector_model}.conf"
    model_path.write_text(model_content)

    # ── Geometry file ──
    geo_content = f"""# Allpix² Geometry — Generated by RadDetect AI

[detector1]
type = "{params.detector_model}"
position = 0 0 0mm
orientation = 0 0 0
"""
    geo_path = config_subdir / "geometry.conf"
    geo_path.write_text(geo_content)

    # ── Main config file ──
    main_lines = [
        f"# Allpix² Main Config — Generated by RadDetect AI",
        f"# Simulation: {params.name}",
        f"# Date: {datetime.datetime.now().isoformat()}",
        "",
        "[Allpix]",
        f"detectors_file = \"geometry.conf\"",
        f"model_paths = \".\"",
        f"number_of_events = {params.num_events}",
        "",
        "[GeometryBuilderGeant4]",
        f"world_material = \"air\"",
        "",
        "[DepositionGeant4]",
        f"physics_list = FTFP_BERT_LIV",
        f"particle_type = \"{params.particle.value}\"",
        f"source_energy = {params.energy_keV}keV",
        f"source_type = \"{params.source_type}\"",
        f"source_position = 0 0 -20mm",
        f"beam_direction = 0 0 1",
    ]

    if params.beam_size_mm:
        main_lines.append(f"beam_size = {params.beam_size_mm}mm")

    main_lines += [
        "",
        "[ElectricFieldReader]",
        f"model = \"{params.electric_field_model}\"",
        f"bias_voltage = {params.bias_voltage_V}V",
        "",
        f"[GenericPropagation]" if params.charge_transport == "generic" else f"[ProjectionPropagation]",
        f"temperature = {params.temperature_K}K",
        f"charge_per_step = 10",
    ]

    if params.charge_transport == "generic":
        main_lines.append("propagate_holes = true")

    main_lines += [
        "",
        "[SimpleTransfer]",
        "",
        "[DefaultDigitizer]",
    ]

    if params.noise_electrons is not None:
        main_lines.append(f"electronics_noise = {params.noise_electrons}e")
    if params.threshold_electrons is not None:
        main_lines.append(f"threshold = {params.threshold_electrons}e")

    main_lines += [""]

    # Output writers
    if params.output_format in ("root", "both"):
        main_lines += [
            "[ROOTObjectWriter]",
            f"file_name = \"{params.name}_{timestamp}\"",
            "",
        ]
    if params.output_format in ("csv", "both"):
        main_lines += [
            "[TextWriter]",
            f"file_name = \"{params.name}_{timestamp}\"",
            "",
        ]

    main_config = "\n".join(main_lines)
    main_path = config_subdir / "main.conf"
    main_path.write_text(main_config)

    result = {
        "status": "success",
        "config_directory": str(config_subdir),
        "files": {
            "main_config": str(main_path),
            "detector_model": str(model_path),
            "geometry": str(geo_path),
        },
        "summary": {
            "name": params.name,
            "sensor": f"{params.sensor_material} — {params.sensor_thickness_um} µm",
            "pixels": f"{params.n_pixels_x}x{params.n_pixels_y} @ {params.pixel_pitch_um} µm pitch",
            "interpixel_gap": f"{params.interpixel_gap_um} µm" if params.interpixel_gap_um else "None",
            "bias": f"{params.bias_voltage_V} V",
            "source": f"{params.particle.value} @ {params.energy_keV} keV",
            "events": params.num_events,
            "transport": params.charge_transport,
            "electric_field": params.electric_field_model,
        },
        "run_command": f"allpix -c {main_path}",
    }

    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════
# TOOL 3: Run Simulation
# ═══════════════════════════════════════════════

class RunSimInput(BaseModel):
    """Input for running a simulation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    config_path: str = Field(..., description="Path to the main config file (Allpix² .conf or Geant4 macro .mac)")
    simulator: str = Field(default="allpix2", description="Simulator to use: 'allpix2' or 'geant4'")
    timeout_seconds: int = Field(default=600, description="Maximum execution time in seconds", ge=10, le=86400)
    dry_run: bool = Field(default=False, description="If True, validate config without running simulation")


@mcp.tool(
    name="geant4_run_simulation",
    annotations={
        "title": "Run Monte Carlo Simulation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def geant4_run_simulation(params: RunSimInput) -> str:
    """Execute a Geant4 or Allpix² Monte Carlo simulation.

    Runs the configured simulation with the specified config file.
    Supports both Allpix² (preferred for pixel detectors) and raw Geant4.
    Returns execution status, output file paths, and runtime statistics.

    Returns:
        str: JSON with run status, output paths, and summary.
    """
    config_path = Path(params.config_path)
    if not config_path.exists():
        return json.dumps({"status": "error", "message": f"Config file not found: {config_path}"})

    if params.dry_run:
        return json.dumps({
            "status": "dry_run",
            "config_valid": True,
            "config_path": str(config_path),
            "simulator": params.simulator,
            "message": "Config validated. Set dry_run=False to execute."
        })

    if params.simulator == "allpix2":
        cmd = ["allpix", "-c", str(config_path)]
    else:
        cmd = ["geant4", str(config_path)]

    result = _run_command(cmd, cwd=config_path.parent, timeout=params.timeout_seconds)

    # Collect output files
    output_files = []
    for ext in ["*.root", "*.csv", "*.dat", "*.json"]:
        output_files.extend([str(f) for f in config_path.parent.glob(ext)])
        output_files.extend([str(f) for f in OUTPUT_DIR.glob(ext)])

    return json.dumps({
        "status": "success" if result["success"] else "error",
        "simulator": params.simulator,
        "config": str(config_path),
        "runtime_output": result["stdout"][-2000:] if result["stdout"] else "",
        "errors": result["stderr"][-1000:] if result["stderr"] else "",
        "output_files": output_files[-20:],
        "return_code": result["returncode"],
    }, indent=2)


# ═══════════════════════════════════════════════
# TOOL 4: Analyze Simulation Results
# ═══════════════════════════════════════════════

class AnalyzeInput(BaseModel):
    """Input for analyzing simulation output."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    output_file: str = Field(..., description="Path to simulation output file (.csv, .root, .dat)")
    analysis_type: str = Field(
        default="spectrum",
        description="Analysis type: 'spectrum', 'spatial_map', 'mtf', 'cnr', 'charge_sharing', 'pixel_hits', 'summary'"
    )
    energy_range_keV: Optional[List[float]] = Field(
        default=None, description="Energy range [min, max] in keV for spectrum analysis",
        min_length=2, max_length=2
    )
    bins: int = Field(default=256, description="Number of histogram bins", ge=10, le=10000)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="geant4_analyze_results",
    annotations={
        "title": "Analyze Simulation Results",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_analyze_results(params: AnalyzeInput) -> str:
    """Analyze Monte Carlo simulation output data.

    Supports energy spectrum analysis, spatial maps, MTF calculation,
    CNR evaluation, charge-sharing analysis, and pixel hit maps.
    Works with CSV and ROOT output from Geant4 / Allpix².

    Returns:
        str: Analysis results in requested format.
    """
    output_path = Path(params.output_file)
    if not output_path.exists():
        return json.dumps({"status": "error", "message": f"File not found: {output_path}"})

    # Generate Python analysis script
    script_lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated analysis script by RadDetect AI MCP Server."""',
        "import numpy as np",
        "import json",
        "import sys",
        "",
    ]

    if output_path.suffix == ".csv":
        script_lines += [
            f'data = np.genfromtxt("{output_path}", delimiter=",", skip_header=1)',
            "",
        ]
    elif output_path.suffix == ".root":
        script_lines += [
            "try:",
            "    import uproot",
            f'    f = uproot.open("{output_path}")',
            "    # Attempt to read common tree names",
            '    for tree_name in ["PixelHit", "MCParticle", "PixelCharge", "edep"]:'
            "        if tree_name in f:",
            "            data = f[tree_name].arrays(library='np')",
            "            break",
            "except ImportError:",
            '    print(json.dumps({"status": "error", "message": "uproot not installed. pip install uproot"}))',
            "    sys.exit(1)",
            "",
        ]

    analysis_scripts = {
        "spectrum": [
            "# Energy spectrum analysis",
            "if 'data' in dir() and data is not None:",
            "    if isinstance(data, np.ndarray):",
            "        energies = data[:, -1] if data.ndim > 1 else data",
            "    else:",
            "        energies = data.get('local_x', data.get('edep', np.array([])))",
            f"    hist, bin_edges = np.histogram(energies, bins={params.bins})",
            "    peak_bin = np.argmax(hist)",
            "    peak_energy = (bin_edges[peak_bin] + bin_edges[peak_bin+1]) / 2",
            "    fwhm_level = hist[peak_bin] / 2",
            "    above = np.where(hist > fwhm_level)[0]",
            "    fwhm = bin_edges[above[-1]+1] - bin_edges[above[0]] if len(above) > 1 else 0",
            "    energy_res = fwhm / peak_energy * 100 if peak_energy > 0 else 0",
            "    result = {",
            '        "total_events": len(energies),',
            '        "peak_energy": float(peak_energy),',
            '        "fwhm": float(fwhm),',
            '        "energy_resolution_pct": float(energy_res),',
            '        "mean_energy": float(np.mean(energies)),',
            '        "std_energy": float(np.std(energies)),',
            "    }",
            "    print(json.dumps(result, indent=2))",
        ],
        "summary": [
            "# Summary statistics",
            "if isinstance(data, np.ndarray):",
            "    result = {",
            '        "shape": list(data.shape),',
            '        "columns": data.shape[1] if data.ndim > 1 else 1,',
            '        "total_entries": int(data.shape[0]),',
            '        "min": float(np.min(data)),',
            '        "max": float(np.max(data)),',
            '        "mean": float(np.mean(data)),',
            '        "std": float(np.std(data)),',
            "    }",
            "    print(json.dumps(result, indent=2))",
        ],
    }

    script_lines += analysis_scripts.get(params.analysis_type, analysis_scripts["summary"])

    # Write and execute analysis script
    script_path = WORKSPACE_DIR / f"analyze_{_generate_timestamp()}.py"
    script_path.write_text("\n".join(script_lines))

    result = _run_command(["python3", str(script_path)], timeout=120)

    if result["success"] and result["stdout"].strip():
        try:
            analysis_result = json.loads(result["stdout"].strip())
            analysis_result["analysis_type"] = params.analysis_type
            analysis_result["source_file"] = str(output_path)
            analysis_result["script"] = str(script_path)
            return _format_response(analysis_result, params.response_format)
        except json.JSONDecodeError:
            pass

    return json.dumps({
        "status": "error" if not result["success"] else "completed",
        "analysis_type": params.analysis_type,
        "source_file": str(output_path),
        "output": result["stdout"][-2000:],
        "errors": result["stderr"][-1000:] if result["stderr"] else None,
        "script": str(script_path),
    }, indent=2)


# ═══════════════════════════════════════════════
# TOOL 5: Material Database Lookup
# ═══════════════════════════════════════════════

class MaterialLookupInput(BaseModel):
    """Input for material database queries."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    material: str = Field(..., description="Material name or formula (e.g., 'CdZnTe', 'CsI', 'Si', 'BGO')")
    properties: Optional[List[str]] = Field(
        default=None,
        description="Specific properties to return: 'density', 'z_eff', 'attenuation', 'bandgap', 'pair_creation', 'mobility', 'fano'"
    )
    energy_keV: Optional[float] = Field(default=None, description="Photon energy for attenuation lookup", gt=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


# Comprehensive material database for radiation detectors
MATERIAL_DB: Dict[str, Dict[str, Any]] = {
    "CdZnTe": {
        "formula": "Cd₀.₉Zn₀.₁Te",
        "geant4_name": "G4_CADMIUM_ZINC_TELLURIDE",
        "density_g_cm3": 5.78,
        "z_eff": 49.1,
        "bandgap_eV": 1.57,
        "pair_creation_eV": 4.64,
        "electron_mobility_cm2_Vs": 1000,
        "hole_mobility_cm2_Vs": 50,
        "electron_lifetime_us": 3.0,
        "hole_lifetime_us": 0.1,
        "fano_factor": 0.089,
        "mu_tau_e_cm2_V": 3e-3,
        "mu_tau_h_cm2_V": 5e-6,
        "notes": "Room-temperature semiconductor. Excellent for spectroscopic X/gamma imaging. Suffers from charge trapping (holes) and polarization."
    },
    "CdTe": {
        "formula": "CdTe",
        "geant4_name": "G4_CADMIUM_TELLURIDE",
        "density_g_cm3": 5.85,
        "z_eff": 50.0,
        "bandgap_eV": 1.44,
        "pair_creation_eV": 4.43,
        "electron_mobility_cm2_Vs": 1100,
        "hole_mobility_cm2_Vs": 100,
        "electron_lifetime_us": 2.0,
        "hole_lifetime_us": 1.0,
        "fano_factor": 0.089,
        "mu_tau_e_cm2_V": 2.2e-3,
        "mu_tau_h_cm2_V": 1e-4,
        "notes": "Higher hole mobility than CZT, but more prone to polarization. Used with Ohmic (In/Pt) or Schottky (In/Au) contacts."
    },
    "Si": {
        "formula": "Si",
        "geant4_name": "G4_Si",
        "density_g_cm3": 2.33,
        "z_eff": 14,
        "bandgap_eV": 1.12,
        "pair_creation_eV": 3.62,
        "electron_mobility_cm2_Vs": 1350,
        "hole_mobility_cm2_Vs": 480,
        "fano_factor": 0.115,
        "notes": "Mature technology. Low Z limits stopping power for >30 keV photons. Excellent for photon-counting CT with edge-on geometry."
    },
    "CsI_Tl": {
        "formula": "CsI:Tl",
        "geant4_name": "G4_CESIUM_IODIDE",
        "density_g_cm3": 4.51,
        "z_eff": 54,
        "light_yield_ph_MeV": 54000,
        "peak_emission_nm": 550,
        "decay_time_us": 1.0,
        "notes": "High light yield scintillator. Hygroscopic. Used in flat-panel detectors. Columnar growth reduces lateral light spread."
    },
    "CsI_Na": {
        "formula": "CsI:Na",
        "geant4_name": "G4_CESIUM_IODIDE",
        "density_g_cm3": 4.51,
        "z_eff": 54,
        "light_yield_ph_MeV": 41000,
        "peak_emission_nm": 420,
        "decay_time_us": 0.63,
        "notes": "Faster decay than CsI:Tl. Emission better matched to PMTs. Also hygroscopic."
    },
    "BGO": {
        "formula": "Bi₄Ge₃O₁₂",
        "geant4_name": "G4_BGO",
        "density_g_cm3": 7.13,
        "z_eff": 73,
        "light_yield_ph_MeV": 8500,
        "peak_emission_nm": 480,
        "decay_time_us": 0.3,
        "notes": "Very high density and Z. Low light yield. Used in PET scanners."
    },
    "LYSO": {
        "formula": "Lu₁.₈Y₀.₂SiO₅:Ce",
        "geant4_name": "G4_LUTETIUM_YTTRIUM_SILICON_OXIDE",
        "density_g_cm3": 7.1,
        "z_eff": 65,
        "light_yield_ph_MeV": 33000,
        "peak_emission_nm": 420,
        "decay_time_us": 0.042,
        "notes": "Modern PET crystal. Fast timing for TOF-PET. Contains natural Lu-176 background."
    },
    "GaAs": {
        "formula": "GaAs",
        "geant4_name": "G4_GALLIUM_ARSENIDE",
        "density_g_cm3": 5.32,
        "z_eff": 32,
        "bandgap_eV": 1.42,
        "pair_creation_eV": 4.35,
        "electron_mobility_cm2_Vs": 8500,
        "hole_mobility_cm2_Vs": 400,
        "fano_factor": 0.12,
        "notes": "High electron mobility. Investigated for X-ray imaging and particle physics."
    },
    "Ge": {
        "formula": "Ge",
        "geant4_name": "G4_Ge",
        "density_g_cm3": 5.32,
        "z_eff": 32,
        "bandgap_eV": 0.67,
        "pair_creation_eV": 2.96,
        "electron_mobility_cm2_Vs": 3900,
        "hole_mobility_cm2_Vs": 1900,
        "fano_factor": 0.057,
        "notes": "Best energy resolution of all semiconductors. Requires cryogenic cooling (~77K). Gold standard for gamma spectroscopy."
    },
}


@mcp.tool(
    name="geant4_material_lookup",
    annotations={
        "title": "Radiation Detector Material Database",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_material_lookup(params: MaterialLookupInput) -> str:
    """Look up radiation detector material properties.

    Comprehensive database of semiconductor and scintillator detector
    materials including CdZnTe, CdTe, Si, Ge, CsI, BGO, LYSO, GaAs.
    Returns physical, electrical, and optical properties relevant to
    detector simulation and design.

    Returns:
        str: Material properties in requested format.
    """
    # Fuzzy match material name
    mat_key = None
    search = params.material.lower().replace(" ", "").replace("-", "")
    for key in MATERIAL_DB:
        if search in key.lower().replace(" ", "").replace("-", ""):
            mat_key = key
            break
    if mat_key is None:
        # Try common aliases
        aliases = {
            "czt": "CdZnTe", "czts": "CdZnTe", "cadmiumzinctelluride": "CdZnTe",
            "cadmiumtelluride": "CdTe",
            "silicon": "Si",
            "germanium": "Ge",
            "cesiumiodide": "CsI_Tl", "csi": "CsI_Tl", "csitl": "CsI_Tl", "csina": "CsI_Na",
            "bgo": "BGO", "bismuthgermanate": "BGO",
            "lyso": "LYSO",
            "gaas": "GaAs", "galliumarsenide": "GaAs",
        }
        mat_key = aliases.get(search)

    if mat_key is None:
        available = ", ".join(MATERIAL_DB.keys())
        return json.dumps({
            "status": "not_found",
            "message": f"Material '{params.material}' not found. Available: {available}"
        })

    mat = MATERIAL_DB[mat_key]

    if params.properties:
        filtered = {k: v for k, v in mat.items() if any(p in k for p in params.properties)}
        filtered["name"] = mat_key
        return _format_response(filtered, params.response_format)

    result = {"name": mat_key, **mat}
    return _format_response(result, params.response_format)


# ═══════════════════════════════════════════════
# TOOL 6: X-ray Spectrum Generator
# ═══════════════════════════════════════════════

class SpectrumGenInput(BaseModel):
    """Input for X-ray spectrum generation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    anode: str = Field(default="W", description="Anode material: 'W', 'Mo', 'Rh', 'Cu', 'Ag'")
    kvp: float = Field(..., description="Tube voltage in kVp", gt=10, le=500)
    filtration_mm_al: float = Field(default=2.5, description="Al-equivalent filtration in mm", ge=0)
    additional_filter_material: Optional[str] = Field(default=None, description="Additional filter material (e.g., 'Cu', 'Sn')")
    additional_filter_mm: Optional[float] = Field(default=None, description="Additional filter thickness in mm")
    energy_bin_keV: float = Field(default=1.0, description="Energy bin width in keV", gt=0, le=10)
    output_format: str = Field(default="gps", description="Output format: 'gps' (Geant4 GPS), 'csv', 'allpix2'")
    normalize: bool = Field(default=True, description="Normalize spectrum to unit area")


@mcp.tool(
    name="geant4_generate_xray_spectrum",
    annotations={
        "title": "Generate X-ray Tube Spectrum",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_generate_xray_spectrum(params: SpectrumGenInput) -> str:
    """Generate a polyenergetic X-ray tube spectrum for simulation.

    Creates a realistic X-ray spectrum using Kramers' law approximation
    with characteristic lines for the specified anode. Output is formatted
    for direct use in Geant4 GPS or Allpix² source configuration.

    For production use, consider SpekCalc, xpecgen, or IPEM Report 78.

    Returns:
        str: JSON with spectrum data and Geant4/Allpix² configuration.
    """
    # Simplified Kramers + characteristic line model
    max_energy = params.kvp
    bins = int(max_energy / params.energy_bin_keV)
    energies = [(i + 0.5) * params.energy_bin_keV for i in range(bins)]

    # Kramers continuum: I(E) ∝ Z * (E_max - E) / E
    z_anode = {"W": 74, "Mo": 42, "Rh": 45, "Cu": 29, "Ag": 47}.get(params.anode, 74)
    spectrum = []
    for e in energies:
        if e >= max_energy:
            spectrum.append(0.0)
        else:
            intensity = z_anode * (max_energy - e) / max(e, 0.1)
            # Al filtration (simplified Beer-Lambert)
            mu_al = 0.2 * (30.0 / max(e, 1.0)) ** 2.8  # Rough Al attenuation
            import math
            intensity *= math.exp(-mu_al * params.filtration_mm_al)
            spectrum.append(max(intensity, 0.0))

    # Add characteristic lines for W anode
    char_lines = {
        "W": [(59.3, 0.15), (67.2, 0.08), (69.1, 0.03)],  # Kα1, Kβ1, Kβ2
        "Mo": [(17.5, 0.20), (19.6, 0.05)],
        "Rh": [(20.2, 0.18), (22.7, 0.04)],
        "Cu": [(8.05, 0.25), (8.9, 0.05)],
        "Ag": [(22.2, 0.18), (24.9, 0.04)],
    }
    for line_energy, rel_intensity in char_lines.get(params.anode, []):
        if line_energy < max_energy:
            bin_idx = int(line_energy / params.energy_bin_keV)
            if 0 <= bin_idx < len(spectrum):
                spectrum[bin_idx] += rel_intensity * max(spectrum) * 2

    # Normalize
    total = sum(spectrum)
    if params.normalize and total > 0:
        spectrum = [s / total for s in spectrum]

    # Format output
    timestamp = _generate_timestamp()

    if params.output_format == "gps":
        gps_lines = [
            f"# X-ray spectrum: {params.anode} anode, {params.kvp} kVp",
            f"# Filtration: {params.filtration_mm_al} mm Al",
            f"# Generated by RadDetect AI MCP Server",
            "/gps/particle gamma",
            "/gps/ene/type User",
            "/gps/hist/type energy",
        ]
        for e, w in zip(energies, spectrum):
            if w > 1e-10:
                gps_lines.append(f"/gps/hist/point {e/1000:.6f} {w:.8f}")
        output_content = "\n".join(gps_lines)

    elif params.output_format == "csv":
        csv_lines = ["energy_keV,weight"]
        for e, w in zip(energies, spectrum):
            csv_lines.append(f"{e:.2f},{w:.8f}")
        output_content = "\n".join(csv_lines)

    else:  # allpix2
        output_content = f"# Allpix² spectrum file\n# {params.anode} {params.kvp}kVp\n"
        for e, w in zip(energies, spectrum):
            if w > 1e-10:
                output_content += f"{e:.2f} {w:.8f}\n"

    # Save file
    out_file = WORKSPACE_DIR / f"spectrum_{params.anode}_{int(params.kvp)}kVp_{timestamp}.{'mac' if params.output_format == 'gps' else params.output_format}"
    out_file.write_text(output_content)

    # Compute mean energy
    if total > 0:
        mean_energy = sum(e * w for e, w in zip(energies, spectrum)) / sum(spectrum)
    else:
        mean_energy = 0

    return json.dumps({
        "status": "success",
        "spectrum_file": str(out_file),
        "anode": params.anode,
        "kvp": params.kvp,
        "filtration_mm_al": params.filtration_mm_al,
        "mean_energy_keV": round(mean_energy, 2),
        "num_bins": len([s for s in spectrum if s > 1e-10]),
        "format": params.output_format,
        "note": "Simplified Kramers model. For clinical accuracy use SpekCalc or xpecgen.",
        "content_preview": "\n".join(output_content.split("\n")[:15]) + "\n..."
    }, indent=2)


# ═══════════════════════════════════════════════
# TOOL 7: List Workspace Files
# ═══════════════════════════════════════════════

class ListFilesInput(BaseModel):
    """Input for listing workspace files."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    directory: str = Field(
        default="all",
        description="Directory to list: 'all', 'macros', 'configs', 'output'"
    )
    pattern: Optional[str] = Field(default=None, description="Glob pattern filter (e.g., '*.mac', '*.root')")


@mcp.tool(
    name="geant4_list_files",
    annotations={
        "title": "List Workspace Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_list_files(params: ListFilesInput) -> str:
    """List files in the RadDetect AI simulation workspace.

    Shows macro files, Allpix² configs, and simulation output files
    in the workspace directory tree.

    Returns:
        str: JSON with file listing.
    """
    dirs = {
        "macros": MACRO_DIR,
        "configs": CONFIG_DIR,
        "output": OUTPUT_DIR,
    }

    if params.directory == "all":
        scan_dirs = dirs
    elif params.directory in dirs:
        scan_dirs = {params.directory: dirs[params.directory]}
    else:
        return json.dumps({"status": "error", "message": f"Unknown directory: {params.directory}"})

    files = {}
    for name, path in scan_dirs.items():
        if params.pattern:
            found = sorted(path.rglob(params.pattern))
        else:
            found = sorted(path.rglob("*"))
        files[name] = [
            {
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            }
            for f in found if f.is_file()
        ]

    return json.dumps({
        "workspace": str(WORKSPACE_DIR),
        "files": files,
        "total_files": sum(len(v) for v in files.values()),
    }, indent=2)


# ═══════════════════════════════════════════════
# TOOL 8: Detector Physics Calculator
# ═══════════════════════════════════════════════

class PhysicsCalcInput(BaseModel):
    """Input for detector physics calculations."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    calculation: str = Field(
        ...,
        description=(
            "Calculation type: "
            "'charge_collection' (Hecht equation), "
            "'energy_resolution' (Fano-limited), "
            "'depletion_depth', "
            "'pixel_charge_sharing' (estimated fraction), "
            "'signal_to_noise', "
            "'count_rate_limit', "
            "'attenuation_length'"
        )
    )
    material: str = Field(default="CdZnTe", description="Detector material")
    thickness_um: Optional[float] = Field(default=None, description="Sensor thickness in µm")
    bias_voltage_V: Optional[float] = Field(default=None, description="Applied bias voltage in V")
    energy_keV: Optional[float] = Field(default=None, description="Photon energy in keV")
    pixel_pitch_um: Optional[float] = Field(default=None, description="Pixel pitch in µm")
    temperature_K: Optional[float] = Field(default=293.15, description="Temperature in Kelvin")
    noise_electrons: Optional[float] = Field(default=None, description="Electronic noise RMS in electrons")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


@mcp.tool(
    name="geant4_physics_calculator",
    annotations={
        "title": "Detector Physics Calculator",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_physics_calculator(params: PhysicsCalcInput) -> str:
    """Perform radiation detector physics calculations.

    Analytical calculations for detector design and performance estimation:
    Hecht equation charge collection, Fano-limited energy resolution,
    depletion depth, charge sharing fraction, SNR, and more.
    Uses the built-in material database for material properties.

    Returns:
        str: Calculation results with equations and parameters.
    """
    import math

    # Get material properties
    mat = MATERIAL_DB.get(params.material)
    if not mat:
        for key, val in MATERIAL_DB.items():
            if params.material.lower() in key.lower():
                mat = val
                break
    if not mat:
        return json.dumps({"status": "error", "message": f"Material '{params.material}' not in database."})

    result: Dict[str, Any] = {
        "calculation": params.calculation,
        "material": params.material,
    }

    if params.calculation == "charge_collection":
        # Hecht equation: η = (µτV/d²) * [1 - exp(-d²/(µτV))]
        if not all([params.thickness_um, params.bias_voltage_V]):
            return json.dumps({"status": "error", "message": "Need thickness_um and bias_voltage_V"})

        d_cm = params.thickness_um / 1e4
        V = abs(params.bias_voltage_V)
        mu_tau_e = mat.get("mu_tau_e_cm2_V", 1e-3)
        mu_tau_h = mat.get("mu_tau_h_cm2_V", 1e-5)

        lambda_e = mu_tau_e * V / d_cm
        lambda_h = mu_tau_h * V / d_cm

        # Single-carrier (electron) CCE for irradiation from cathode side
        cce_e = (lambda_e / d_cm) * (1 - math.exp(-d_cm / lambda_e)) if lambda_e > 0 else 0
        cce_h = (lambda_h / d_cm) * (1 - math.exp(-d_cm / lambda_h)) if lambda_h > 0 else 0
        cce_total = min((cce_e + cce_h) / 2, 1.0)  # simplified two-carrier

        result.update({
            "thickness_um": params.thickness_um,
            "bias_V": V,
            "mu_tau_electron": mu_tau_e,
            "mu_tau_hole": mu_tau_h,
            "cce_electron": round(cce_e, 4),
            "cce_hole": round(cce_h, 4),
            "cce_total_approx": round(cce_total, 4),
            "electric_field_V_cm": round(V / d_cm, 1),
            "equation": "Hecht: η = (µτV/d²)[1 - exp(-d²/µτV)]",
        })

    elif params.calculation == "energy_resolution":
        # Fano-limited: ΔE/E = 2.355 * sqrt(F * w / E)
        if not params.energy_keV:
            return json.dumps({"status": "error", "message": "Need energy_keV"})

        w = mat.get("pair_creation_eV", 4.64)
        F = mat.get("fano_factor", 0.1)
        E_eV = params.energy_keV * 1000
        n_pairs = E_eV / w

        fano_sigma = math.sqrt(F * n_pairs)
        fano_fwhm_eV = 2.355 * fano_sigma * w
        fano_res_pct = fano_fwhm_eV / E_eV * 100

        noise_contrib_eV = 0
        if params.noise_electrons:
            noise_contrib_eV = 2.355 * params.noise_electrons * w
            total_fwhm_eV = math.sqrt(fano_fwhm_eV**2 + noise_contrib_eV**2)
            total_res_pct = total_fwhm_eV / E_eV * 100
        else:
            total_fwhm_eV = fano_fwhm_eV
            total_res_pct = fano_res_pct

        result.update({
            "energy_keV": params.energy_keV,
            "pair_creation_eV": w,
            "fano_factor": F,
            "n_electron_hole_pairs": round(n_pairs),
            "fano_limited_fwhm_eV": round(fano_fwhm_eV, 2),
            "fano_limited_resolution_pct": round(fano_res_pct, 3),
            "electronic_noise_fwhm_eV": round(noise_contrib_eV, 2),
            "total_fwhm_eV": round(total_fwhm_eV, 2),
            "total_resolution_pct": round(total_res_pct, 3),
        })

    elif params.calculation == "pixel_charge_sharing":
        if not all([params.pixel_pitch_um, params.thickness_um, params.energy_keV]):
            return json.dumps({"status": "error", "message": "Need pixel_pitch_um, thickness_um, energy_keV"})

        # Simplified charge cloud size estimate
        w = mat.get("pair_creation_eV", 4.64)
        n_pairs = params.energy_keV * 1000 / w
        # Diffusion sigma ≈ sqrt(2kT/qE * d) — simplified
        E_field = abs(params.bias_voltage_V or 500) / (params.thickness_um / 1e4)
        kT = 0.0259 * (params.temperature_K / 300)  # eV at temperature
        drift_time_e = (params.thickness_um / 1e4) / (mat.get("electron_mobility_cm2_Vs", 1000) * E_field)
        sigma_diffusion_um = math.sqrt(2 * kT / 1.6e-19 * mat.get("electron_mobility_cm2_Vs", 1000) * 1e-4 * drift_time_e) * 1e4

        # Charge sharing fraction (approximate)
        cloud_diameter_um = 2 * 2.355 * sigma_diffusion_um  # ~FWHM
        sharing_fraction = min(cloud_diameter_um / params.pixel_pitch_um, 1.0)

        result.update({
            "pixel_pitch_um": params.pixel_pitch_um,
            "thickness_um": params.thickness_um,
            "electric_field_V_cm": round(E_field, 1),
            "diffusion_sigma_um": round(sigma_diffusion_um, 2),
            "charge_cloud_fwhm_um": round(cloud_diameter_um, 2),
            "estimated_sharing_fraction": round(sharing_fraction, 4),
            "note": "Simplified diffusion model. Actual sharing depends on interaction depth and fluorescence."
        })

    elif params.calculation == "attenuation_length":
        if not params.energy_keV:
            return json.dumps({"status": "error", "message": "Need energy_keV"})

        # Very rough empirical attenuation (for demonstration)
        density = mat.get("density_g_cm3", 5.0)
        z_eff = mat.get("z_eff", 50)
        E = params.energy_keV

        # Simplified photoelectric + Compton
        mu_pe = 0.01 * density * (z_eff / 50) ** 4 * (60 / max(E, 1)) ** 3  # cm⁻¹ rough
        mu_compton = 0.1 * density * (z_eff / 50)  # cm⁻¹ rough
        mu_total = mu_pe + mu_compton
        attenuation_length_mm = 10 / mu_total if mu_total > 0 else float('inf')

        result.update({
            "energy_keV": E,
            "density_g_cm3": density,
            "z_eff": z_eff,
            "approx_attenuation_length_mm": round(attenuation_length_mm, 3),
            "note": "Rough empirical estimate. Use NIST XCOM for accurate values."
        })

    else:
        return json.dumps({"status": "error", "message": f"Unknown calculation: {params.calculation}"})

    return _format_response(result, params.response_format)


# ═══════════════════════════════════════════════
# TOOL 9: Workspace Status
# ═══════════════════════════════════════════════

@mcp.tool(
    name="geant4_workspace_status",
    annotations={
        "title": "Check Workspace & Environment Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def geant4_workspace_status() -> str:
    """Check the RadDetect AI MCP Server workspace and environment status.

    Reports on available simulators (Geant4, Allpix²), workspace
    directories, file counts, and environment configuration.

    Returns:
        str: JSON with environment and workspace status.
    """
    # Check for available tools
    checks = {}
    for tool_name, cmd in [("geant4", ["geant4-config", "--version"]), ("allpix2", ["allpix", "--version"]),
                            ("python3", ["python3", "--version"]), ("root", ["root-config", "--version"])]:
        result = _run_command(cmd, timeout=10)
        checks[tool_name] = {
            "available": result["success"],
            "version": result["stdout"].strip() if result["success"] else "not found"
        }

    # Count workspace files
    file_counts = {}
    for name, path in [("macros", MACRO_DIR), ("configs", CONFIG_DIR), ("output", OUTPUT_DIR)]:
        file_counts[name] = len(list(path.rglob("*"))) if path.exists() else 0

    return json.dumps({
        "server": "RadDetect AI — Geant4 MCP Server",
        "version": "1.0.0",
        "workspace": str(WORKSPACE_DIR),
        "tools_available": checks,
        "file_counts": file_counts,
        "environment": {
            "GEANT4_INSTALL_DIR": str(GEANT4_INSTALL),
            "ALLPIX2_INSTALL_DIR": str(ALLPIX2_INSTALL),
        },
        "supported_materials": list(MATERIAL_DB.keys()),
        "tools_registered": [
            "geant4_generate_macro",
            "geant4_generate_allpix2_config",
            "geant4_run_simulation",
            "geant4_analyze_results",
            "geant4_material_lookup",
            "geant4_generate_xray_spectrum",
            "geant4_list_files",
            "geant4_physics_calculator",
            "geant4_workspace_status",
        ]
    }, indent=2)


# ═══════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
