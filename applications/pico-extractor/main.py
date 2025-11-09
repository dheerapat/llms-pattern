import typer
from pdf_extractor import extract_text_from_pdf
from pico_agents import (
    get_comparison,
    get_intervention,
    get_outcome,
    get_population,
)
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn


def main(
    pdf_path: str = typer.Argument(..., help="Path to the clinical research PDF file."),
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Extracting text from PDF...", total=None)
        try:
            text = extract_text_from_pdf(pdf_path)
            if not text:
                print(
                    "[bold red]Error: Could not extract text from the PDF. It may be empty or image-based.[/bold red]"
                )
                raise typer.Exit(1)
        except FileNotFoundError as e:
            print(f"[bold red]Error: {e}[/bold red]")
            raise typer.Exit(1)

        print(f"Successfully extracted {len(text)} characters of text.")

        # Run PICO agents
        progress.add_task("Analyzing Population (P)...", total=None)
        population = get_population(text)

        progress.add_task("Analyzing Intervention (I)...", total=None)
        intervention = get_intervention(text)

        progress.add_task("Analyzing Comparison (C)...", total=None)
        comparison = get_comparison(text)

        progress.add_task("Analyzing Outcome (O)...", total=None)
        outcome = get_outcome(text)

    print("\n[bold green]PICO Extraction Complete:[/bold green]")

    print("\n[bold blue]P - Population:[/bold blue]")
    print(population.model_dump_json(indent=2))

    print("\n[bold blue]I - Intervention:[/bold blue]")
    print(intervention.model_dump_json(indent=2))

    print("\n[bold blue]C - Comparison:[/bold blue]")
    print(comparison.model_dump_json(indent=2))

    print("\n[bold blue]O - Outcome:[/bold blue]")
    print(outcome.model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
