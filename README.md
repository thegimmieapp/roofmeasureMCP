# RoofMeasure MCP

Free EagleView-style satellite roof measurements plus modular Xactimate-style
estimating, packaged as an MCP server and CLI for any LLM or agent.

**What it measures** (from Google Solar API 3D elevation data):

- Total roof area (sq ft and squares), with and without waste (0-32% waste table, squares rounded up to 1/3 SQ)
- Facet count, per-facet area, pitch, and slope direction
- Predominant pitch and areas-per-pitch table
- Ridges, hips, valleys, rakes, and eaves lengths (slope-corrected)
- Drip edge (eaves + rakes), steep-slope area split for labor surcharges
- Suggested waste factor from roof complexity

**What it generates**:

- EagleView-style measurement report (Markdown)
- Xactimate-style insurance estimate (.docx) with line items, material sales
  tax, per-structure summaries, tax recap, and grand total. Fully modular:
  company name/logo, estimator, homeowner, claim info, pricing rules, tax
  rate, waste %, component counts, and O&P toggle are all configurable.

## Setup

1. **Get a free Google Cloud API key** with the **Geocoding API** and **Solar
   API** enabled ([console.cloud.google.com](https://console.cloud.google.com)).
   The $200/month free credit covers roughly 1,000 roof lookups.

2. **Install**:

   ```bash
   pip install git+https://github.com/thegimmieapp/roofmeasureMCP.git
   # or from a local clone:
   pip install -e .
   ```

3. **Set your key**:

   ```bash
   export GOOGLE_MAPS_API_KEY="your_key_here"
   ```

## Use as an MCP server (Claude Desktop, Claude Code, Cursor, etc.)

Add to your MCP config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "roofmeasure": {
      "command": "roofmeasure-mcp",
      "env": {
        "GOOGLE_MAPS_API_KEY": "your_key_here",
        "ROOFMEASURE_OUT_DIR": "/path/for/reports",
        "ROOFMEASURE_LOGO": "/path/to/company_logo.png"
      }
    }
  }
}
```

Claude Code: `claude mcp add roofmeasure -e GOOGLE_MAPS_API_KEY=your_key -- roofmeasure-mcp`

### Tools

| Tool | Purpose |
|---|---|
| `measure_roof(address)` | Full JSON measurements |
| `generate_roof_report(address, ...)` | EagleView-style Markdown report file |
| `generate_xactimate_estimate(address, homeowner, date_of_loss, ...)` | Xactimate-style .docx estimate |

The estimate tool enforces contractor intake: if homeowner name or date of
loss are missing it returns `needs_info` so the agent asks the user before
generating.

## Use from the command line

```bash
roofmeasure measure "3708 Ebony Hollow Pass, Austin, TX 78739"
roofmeasure report  "3708 Ebony Hollow Pass, Austin, TX 78739" -o report.md
roofmeasure estimate "3708 Ebony Hollow Pass, Austin, TX 78739"   # prompts for homeowner, date of loss, claim #
```

## How it works

1. Geocodes the address, pulls Google Solar API `buildingInsights` and
   `dataLayers` (0.1 m/px digital surface model + roof mask GeoTIFFs).
2. Segments the DSM into planar facets (region growing on slope + aspect).
3. Classifies every facet boundary: shared boundaries become ridges, hips, or
   valleys (by relative elevation and facing); perimeter boundaries become
   eaves or rakes (by edge direction vs the facet's downslope direction).
   Sloped edge lengths are slope-corrected in 3D.
4. If high-resolution DSM data is not available for an address, it falls back
   to roof-segment statistics and clearly flags edge lengths as ESTIMATED.

## Accuracy notes

This tool is free and uses the best publicly available elevation data. It is
strong on areas, pitch, and squares; edge classification is measured from the
DSM but can be affected by tree cover, imagery age, and resolution. It does
not carry an accuracy guarantee - field verify before ordering material, as
you would with any measurement report.

## Estimating defaults (all overridable)

- Architectural laminated shingles + synthetic 30# felt only
- 1:1 replacement of existing components
- ~$550/SQ blended target across core roofing components (shingle unit price back-solved)
- Steep-roof labor surcharges at 8/12-9/12 and 10/12+ tiers from the measured pitch table
- Material sales tax applied to the material fraction of each line item
- Overhead & Profit OFF by default (`include_op: true` to add 10/10)

MIT license.
