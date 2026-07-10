"""RoofMeasure CLI.

  roofmeasure measure "123 Main St, Austin, TX 78701"
  roofmeasure report "123 Main St, Austin, TX 78701" -o report.md
  roofmeasure estimate "123 Main St, ..." --homeowner "Jane Doe" --date-of-loss 5/14/2026
"""

from __future__ import annotations

import json

import typer

from . import pipeline, report as report_mod

app = typer.Typer(help="Free satellite roof measurements + Xactimate-style estimating.")


@app.command()
def measure(address: str):
    """Print full roof measurements as JSON."""
    m = pipeline.measure_address(address)
    typer.echo(m.to_json())


@app.command()
def report(
    address: str,
    out: str = typer.Option("", "-o", "--out", help="Output .md path"),
    company: str = typer.Option("Stronghouse Solutions", help="Company name on the report"),
):
    """Generate an EagleView-style report (PDF with labeled diagrams + Markdown)."""
    m, detail, rgb = pipeline.measure_address_full(address)
    md = report_mod.render_markdown(m, company=company)
    if not out:
        out = f"Roof_Report_{address.split(',')[0].replace(' ', '_')}.md"
    with open(out, "w") as f:
        f.write(md)
    if detail is not None:
        from .diagrams import render_pdf_report

        pdf = out.rsplit(".", 1)[0] + ".pdf"
        render_pdf_report(m, detail, pdf, rgb=rgb, company=company)
        typer.echo(f"Saved {pdf}")
    typer.echo(f"Saved {out}")
    typer.echo(f"Total: {m.total_area_sqft:,.0f} sq ft ({m.squares:.2f} SQ), "
               f"{m.facet_count} facets, predominant pitch {m.predominant_pitch} [{m.method}]")


@app.command()
def estimate(
    address: str,
    homeowner: str = typer.Option(..., prompt="Homeowner name"),
    phone: str = typer.Option("", prompt="Homeowner phone (blank if unknown)"),
    date_of_loss: str = typer.Option(..., prompt="Date of loss (m/d/yyyy)"),
    claim_number: str = typer.Option("TBD", prompt="Claim number (TBD if unknown)"),
    policy_number: str = typer.Option("TBD", prompt="Policy number (TBD if unknown)"),
    tax_rate: float = typer.Option(0.0825, help="Combined local sales tax rate (decimal)"),
    waste_pct: float = typer.Option(0.0, help="Waste % (0 = auto-suggest)"),
    components: str = typer.Option("{}", help='JSON component counts, e.g. {"pipe_jacks":3}'),
    company_name: str = typer.Option("Stronghouse Solutions"),
    company_city: str = typer.Option(""),
    include_op: bool = typer.Option(False, help="Include 10/10 O&P (off by default)"),
):
    """Generate an Xactimate-style .docx estimate (prompts for missing intake info)."""
    from .server import generate_xactimate_estimate

    result = generate_xactimate_estimate(
        address=address,
        homeowner=homeowner,
        homeowner_phone=phone,
        date_of_loss=date_of_loss,
        claim_number=claim_number,
        policy_number=policy_number,
        tax_rate=tax_rate,
        waste_pct=waste_pct,
        components_json=components,
        company_name=company_name,
        company_city=company_city,
        include_op=include_op,
    )
    typer.echo(json.dumps(json.loads(result), indent=2))


@app.command()
def serve():
    """Run the MCP server (stdio)."""
    from .server import main as serve_main

    serve_main()


if __name__ == "__main__":
    app()