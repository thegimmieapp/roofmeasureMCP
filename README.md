# RoofMeasure MCP

Free EagleView-style satellite roof measurements plus modular Xactimate-style
estimating, packaged as an MCP server and CLI for any LLM or agent.

**What it measures** (from Google Solar API 0.1 m elevation data):

- Total roof area (sq ft and squares) from fitted roof planes, per-facet area, pitch and slope direction
- Ridges, hips, valleys, rakes and eaves: every crease is the analytic
  intersection of two measured roof planes with its true sloped length;
  rakes are slope-corrected; eaves are level
- Predominant pitch and areas-per-pitch table
- EagleView-convention waste table (3/12 and steeper area only, columns
  around the suggested factor, squares rounded up to 1/3) and suggested waste
  factor calibrated to EagleView reports
- Structure complexity (Simple / Normal / Complex), drip edge, steep-slope split
- Tree-canopy confidence rating (HIGH / MEDIUM / LOW) and Google's own
  whole-roof area as an independent cross-check
- Detached roofs nearby (garage, shed) reported, and measured as extra
  structures with `include_outbuildings=true`

**Accuracy** against six EagleView Premium Reports (Springfield, MO; primary
structure): total area within 1% on simple roofs and +5 to +9% on complex
ones; rakes within 1 to 9%; eaves within 4 to 8% (15% low where tree canopy
hides a side); ridges+hips within 2 to 15% except where crest blur in the
elevation model adds a facet (Fred Eagles: +24%). The engine follows
EagleView's conventions: an upper roof's edge over a lower roof counts as
an interior eave or rake, the wall junction as flashing / step flashing,
waste on 3/12-and-steeper area only. Roofs bordered by tree canopy are
flagged LOW confidence and are a lower bound. Our outline matches
EagleView's drawn outline within about one foot on the canopy-free houses
(verified by overlaying the vectors extracted from their PDFs, see
`tests/ev_extract.py`).

**Geometry rules** (src/roofmeasure/regularize.py): measured planes are
snapped to how US homes are framed before any line is drawn. Every pitched
plane faces one of the building's four axis directions (45 degrees only
for a bay-sized facet); pitch is a whole x/12 and planes within 1/12 of the
wing's pitch take the wing's pitch; a strip between two parallel planes at
different heights is a connector facet (hip + valley), a strip between
planes at the same height is a blurred shoulder and dissolves. Hips and
valleys between equal-pitch planes therefore run at exactly 45 degrees in
plan and ridges sit centred, as framed. No gambrels: a same-facing pitch
change with no wall between the two parts (break parallel to the ridge,
flatter plane above) is one plane. Two same-facing, same-pitch neighbours
offset by less than 0.6 m with a continuous surface across their boundary
are one plane (a blurred shoulder); a jump in the elevation model there is
a wall and stays a step. A side-by-side section at a different pitch on the
same slope (Tony Russell's 7/12 insert on a 10/12 roof) is kept and its
sides are walls (step flashing), as EagleView draws it.

**Framing plan** (`src/roofmeasure/roofgraph.py`, v0.5.0): the line
generator for every report. Instead of intersecting fitted planes, it
generates the ridge / hip / valley graph from the eave line the way a
framer lays it out: every plane rises from its eave at one pitch, so hips
run at 45 degrees from outside corners, valleys run from inside corners to
the eaves, ridges sit centred over each wing, a narrower wing's ridge ends
where its valleys meet and ties to the main ridge with a short hip (the
straight skeleton of the eave polygon, computed with bpypolyskel). Every
line ends where it meets another line, no line is drawn twice, and no hip
starts anywhere but a corner, by construction. The elevation model supplies
only what geometry cannot: the pitch (consensus of the fitted planes),
gable ends, lower sections (porches, carports) and a fit check.

Gable ends are hypotheses, not verdicts. Two detectors propose them: the
elevation profile one metre inside an eave (a tent whose flanks have the
roof pitch; the flank fit gives the gable's width and position, so a gable
can occupy part of a straight eave when a wing's end wall is flush with
the eave beside it) and the fitted planes (a plane sloping along the edge
is a rake). Every combination of the candidates is built and the plan the
elevation data agrees with best is kept, with the plain hip plan always in
the running. A flush gable produces the valley a framer would cut from the
gable corner up to the ridge.

The plan is used when the measured planes support it: at least 70% of the
framed ridge/hip/valley length lies within 1 m of a measured plane crease
and the plan does not invent more than 1.6x the measured crease length
(or the elevation residual is within 24 in at the 90th percentile). Its
confidence is reported (confirmed, or MEDIUM with the residual). On a
MEDIUM plan, where the measured planes contain more than 15% more crease
length than the framed plan draws, the larger ridge/hip/valley totals are
reported for ordering and the note says so; the diagram stays framed.
Where the planes do not support the plan (several levels and pitches the
single-pitch model does not describe), lines come from the measured plane
intersections cleaned to the same construction rules
(`wireframe.sanitize_creases`: duplicate near-parallel creases removed,
collinear ridge pieces merged, hips moved to the nearest outline corner or
dropped) and then the topology repair below. Dependency: bpypolyskel (GPL,
pure Python, installed from GitHub). Blender's `mathutils` C extension is
not required: roofmeasure ships the small subset bpypolyskel needs
(`src/roofmeasure/_mathutils.py`, double precision), so no compiler is
needed on macOS or in the Docker image.

Calibration (six EagleView reports, Springfield MO, measured values):
Jenny R142/142 ft, hips 0/0, rakes 158/160; Joe hips 397/401, valleys
214/239; Tony ridge+hip 172/176, valleys 130/129; Fred and Andy are MEDIUM
plans (Fred hips 278/216 after the ordering check, Andy 153/189); Lynda is
canopy-bound and low confidence. 3669 W Maplewood (the report that
prompted v0.5): framed plan confirmed within 20 in, 2 gables, R73 H161
V120, replacing a plane wireframe that drew duplicate hips and valleys.

**Tree canopy** (v0.5.1): the footprint follows the canopy where trees
overhang the roof, and a tree line is never a roof line. An outline edge is
untrusted when it runs off both building axes or when tall vegetation
stands right outside it; each run of untrusted edges between two trusted
edges is replaced by the rectilinear eave line (`wireframe.complete_canopy_runs`):
extend the trusted eave, follow the side at the outermost extent a trusted
corner establishes, return to the next trusted edge. A cut-off corner
becomes the corner; a whole side pushed inward by canopy is pushed back out.
The completed strip must be hidden by something standing above the ridge
(a tree beside the house is not hidden roof), may not exceed 45% of the
visible area, and its elevation is never used for pitch, gable profiles or
the fit (`generate_roof(known=...)`); the plan-support test ignores lines
under canopy. Any remaining off-axis edge is replaced by the two axis
edges through the outside corner (`roofgraph.rectilinearize`). The report
states how much plan area was recovered.

**Topology rules** (`wireframe.enforce_topology`): every ridge, hip and
valley must end where it meets another line. A hip starts at an outline
corner and ends at the point that begins a ridge; a valley starts at the
eave and ends at a ridge or hip junction; a ridge ends at a hip pair, a
valley pair or a gable end. A crease end that dangles inside the roof is
extended along its own line (through roof pixels only) until it meets
another crease or the outline, up to 3 m (4.5 m for a valley reaching for
its eave); a short crease that still touches nothing is dropped. The
report notes how many ends were repaired.

**Eave and rake ordering margin**: outline edges are the least certain
lines in satellite elevation data (overhangs, canopy, eave-vs-rake
classification at walls), and short drip edge or starter is the expensive
error. Reported eave and rake totals are the measured lengths times
`model.EAVE_RAKE_SAFETY` (1.15). The measured values are kept in
`eaves_measured_ft` / `rakes_measured_ft`, printed on the Length Diagram
and Report Summary, and the margin is stated in the notes. Diagram segment
labels are measured lengths. Ridges, hips, valleys and areas carry no
margin.

**What it generates**:

- EagleView-layout measurement report PDF: cover, aerial, Length / Pitch /
  Area / Notes diagrams (EagleView colors, rotated segment labels), per-
  structure Report Summary (areas per pitch, complexity, waste table,
  lengths), notice page; plus a Markdown version
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
| `measure_roof(address, include_outbuildings=False)` | Full JSON measurements |
| `generate_roof_report(address, company, contact, phone, email, company_address, claim_number, include_outbuildings)` | EagleView-layout PDF + Markdown report |
| `generate_xactimate_estimate(address, homeowner, date_of_loss, ...)` | Xactimate-style .docx estimate |

The estimate tool enforces contractor intake: if homeowner name or date of
loss are missing it returns `needs_info` so the agent asks the user before
generating.

## Use from any chat: run it as a remote connector

The plugin above only exists while Claude Desktop is open on the machine
that has it installed. To use roof-measure from any claude.ai chat (web,
desktop, mobile), run the same server over HTTPS and add it as a custom
connector.

1. Deploy (Fly.io shown; any host that runs a Docker container works):

       fly launch --no-deploy --copy-config --name roofmeasure-mcp
       fly volumes create reports --size 1 --region dfw
       fly secrets set GOOGLE_MAPS_API_KEY=... \
           ROOFMEASURE_PATH_SECRET=$(openssl rand -hex 16) \
           ROOFMEASURE_PUBLIC_URL=https://roofmeasure-mcp.fly.dev
       fly deploy

   `fly secrets list` shows the secret name; keep the value you generated.
   `.github/workflows/fly-deploy.yml` redeploys on every push to main once
   a `FLY_API_TOKEN` repository secret exists (`fly tokens create deploy`).

2. In claude.ai: Settings > Connectors > Add custom connector, name it
   RoofMeasure and paste `https://roofmeasure-mcp.fly.dev/<secret>/mcp`.
   No OAuth is configured, so the unguessable path segment is the access
   control: anyone with the full URL can run measurements on your Google
   key. Rotate the secret (`fly secrets set ROOFMEASURE_PATH_SECRET=...`)
   if it leaks.

3. Enable the connector in a chat. Tool results now carry
   `report_pdf_url` / `estimate_url` download links served from the same
   app (`/<secret>/files/<name>`), since the files are no longer on your
   own disk. `GET /health` is a public liveness check.

Entry point: `roofmeasure-mcp-http` (Streamable HTTP, stateless, JSON
responses). Environment: `PORT`, `GOOGLE_MAPS_API_KEY`,
`ROOFMEASURE_PATH_SECRET`, `ROOFMEASURE_PUBLIC_URL`, `ROOFMEASURE_OUT_DIR`.

## LotMeasure: the lot for landscape design (v0.6.0)

Two more MCP tools on the same server, plus a JSON API for websites.

- `measure_lot(address, include_sun=True)`: lot line (county GIS parcel
  polygon) with area in sq ft and acres and the county's recorded size; the
  house footprint from RoofMeasure; front / back / left / right yard zones
  (left and right as seen from the street), each with area, surface
  breakdown (lawn and planting, hardscape, tree canopy, pool, bare soil or
  mulch, detached structure, unknown), the setback from the house wall to the
  lot line, the largest clear plantable rectangle and sun classes; trees with
  location, crown diameter, height and whether they are within 3 m of the
  house; totals; confidence notes.
- `generate_lot_plan(address, brand="Stronghouse Solutions", prepared_for="")`:
  plan PNG + PDF (`plan_png_url`, `plan_pdf_url`). `brand="Dream Landscaper"`
  gives the neutral DL header.

Data sources, in order: county / state ArcGIS parcel layers (seeded and
verified in `src/roofmeasure/parcels_registry.json` for Greene MO, Christian
MO (City of Nixa area), Travis TX, Williamson TX, Santa Rosa and Escambia FL,
plus the Florida statewide layer; other counties are discovered on ArcGIS
Online and cached as unverified); Regrid when `REGRID_API_KEY` is set;
otherwise `lot_source: "none"` and no lot size is reported. Street frontage:
OpenStreetMap road centreline when Overpass answers, else the lot line with
no neighbouring parcel across it (driveway breaks ties), else the line the
driveway reaches. Sun: Solar API hourly shade (June and December) and annual
flux; plantable ground is full sun at 6 h or more of June sun, part sun 3 to
6 h, shade under 3 h. Solar rasters are UTM (EPSG:326xx); `geo.TileGeo` maps
lat/lng to pixels and back without pyproj.

JSON API (HTTP mode), header `X-API-Key: <LOTMEASURE_API_KEY>`:

    GET /<secret>/api/lot?address=...&sun=1&plan=1&brand=Dream%20Landscaper&refresh=0
    GET /<secret>/api/roof?address=...
    GET /<secret>/api/health

Results are cached by address for 24 h on the volume. Set the key with
`fly secrets set LOTMEASURE_API_KEY=$(openssl rand -hex 24)`.

Calibration: `GOOGLE_MAPS_API_KEY=... python tests/fetch_lot_cache.py <dir>`
caches raw GeoTIFFs; `python tests/calibrate_lot.py <dir> [plan_dir]`
compares lot areas with the county record and draws plans.

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
DSM but can be affected by tree cover, imagery age, and resolution. Current
calibration on five canopy-free EagleView reports (main structure, measured
values before the eave/rake margin): area within -1% to +9%; hips within
+5% to +9% and valleys within -3% to +25% on three of five; one roof (Fred
Eagles) has a crest the elevation model blurs into extra facets, giving
hips +30% and valleys +71%; eaves within -7% to +4%; rakes -22% to +7%
(so the +15% margin still leaves one roof short on rakes). It does
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

## Calibration against EagleView

`tests/fetch_cache.py` downloads the Solar API rasters for a set of
addresses once (needs `GOOGLE_MAPS_API_KEY`); `tests/calibrate.py` then runs
the engine offline and prints every measurement next to the EagleView value.
Add a report's numbers to `TRUTH` in `calibrate.py` to extend the set.
Synthetic roofs with analytic truth live in `tests/synth.py` (`pytest`).
