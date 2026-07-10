# RoofMeasure MCP - User Setup Guide

Connect RoofMeasure to Claude (or any MCP-capable agent) with your own Google
Cloud credentials. Takes about 10 minutes; typical usage costs $0.

## Step 1: Create your Google Cloud API key

RoofMeasure uses Google's Geocoding API (address lookup) and Solar API (3D
roof geometry). You need your own free API key:

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and sign
   in with any Google account.
2. Create a project (or use an existing one) via the project selector at the
   top of the page.
3. **Enable billing** on the project (Billing > Link a billing account). A
   card is required, but Google gives every account a $200/month free credit
   for Maps Platform APIs; roughly 1,000 roof lookups fit inside it. Typical
   contractor usage never gets billed.
4. Enable the two APIs: go to **APIs & Services > Library**, search for and
   enable **Geocoding API** and **Solar API**.
5. Create the key: **APIs & Services > Credentials > Create Credentials > API
   key**. Copy it.
6. Recommended: click the key, and under **API restrictions** select
   "Restrict key" and check only Geocoding API and Solar API. Leave "service
   account" binding unchecked; it is not needed.

## Step 2: Connect the MCP server

Requirement: [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS, or
`pipx install uv`). The server auto-installs from GitHub via `uvx`; nothing
else to install.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings > Developer > Edit Config):

```json
{
  "mcpServers": {
    "roofmeasure": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/thegimmieapp/roofmeasureMCP.git", "roofmeasure-mcp"],
      "env": {
        "GOOGLE_MAPS_API_KEY": "YOUR_API_KEY_HERE",
        "ROOFMEASURE_OUT_DIR": "/path/where/reports/should/save"
      }
    }
  }
}
```

Restart Claude Desktop after saving.

### Claude Code (CLI)

```bash
claude mcp add roofmeasure \
  -e GOOGLE_MAPS_API_KEY=YOUR_API_KEY_HERE \
  -e ROOFMEASURE_OUT_DIR="$HOME/RoofMeasure Reports" \
  -- uvx --from git+https://github.com/thegimmieapp/roofmeasureMCP.git roofmeasure-mcp
```

### Cowork (plugin)

Install the `roof-measure.plugin` file, then edit the plugin's `.mcp.json`
and replace the `GOOGLE_MAPS_API_KEY` value with your own key.

### Cursor / other MCP clients

Any client that supports stdio MCP servers works with the same command:
`uvx --from git+https://github.com/thegimmieapp/roofmeasureMCP.git roofmeasure-mcp`
with the `GOOGLE_MAPS_API_KEY` environment variable set.

### Without uv (plain pip)

```bash
pip install git+https://github.com/thegimmieapp/roofmeasureMCP.git
```

Then use `roofmeasure-mcp` as the command (no args) in any config above. This
also installs the `roofmeasure` CLI for terminal use.

## Step 3: Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_MAPS_API_KEY` | Yes | Your Google Cloud key (Geocoding + Solar APIs enabled) |
| `ROOFMEASURE_OUT_DIR` | No | Folder for reports/estimates (default `~/RoofMeasure Reports`) |
| `ROOFMEASURE_LOGO` | No | Path to a PNG logo for estimate letterhead |

## Step 4: Try it

Ask your agent:

- "Measure the roof at 123 Main St, Dallas, TX 75201"
- "Create a roof measurement report for [address]" (PDF with aerial image and labeled length/pitch/area diagrams)
- "Create an Xactimate estimate for [address]" (the agent will ask for homeowner and date of loss)

Company branding on estimates defaults to Stronghouse Solutions; pass your
own company name, tagline, and city to the estimate tool (or ask your agent
to use your company) to rebrand every document.

## Troubleshooting

- **REQUEST_DENIED / "must enable Billing"**: billing is not linked on the
  project the key belongs to, or the APIs are not enabled. Check the project
  selector; changes take a few minutes to propagate.
- **"Solar building data unavailable"**: Google has no 3D coverage for that
  address (rural areas mostly). The tool falls back to estimated edges and
  flags them.
- **Edge lengths flagged approximate**: dense tree cover hides parts of the
  roof from satellites. Areas and pitch stay accurate; field verify eave/hip
  lengths before ordering material.
