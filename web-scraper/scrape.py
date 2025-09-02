#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import typer
from typing import Optional

app = typer.Typer(help="Web scraper for extracting content from NCBI and similar pages")


@app.command()
def scrape(
    url: str = typer.Argument(..., help="URL to scrape"),
    user_agent: Optional[str] = typer.Option(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "--user-agent",
        "-ua",
        help="Custom User-Agent string",
    ),
    timeout: Optional[int] = typer.Option(
        30, "--timeout", "-t", help="Request timeout in seconds"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Scrape content from a web page and extract specific elements.

    This script looks for:
    - div with class 'document'
    - div with class 'bk_tt'
    - span with class 'bk_cite_avail'
    """

    if verbose:
        typer.echo(f"🌐 Fetching URL: {url}")
        typer.echo(f"⏱️  Timeout: {timeout}s")

    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
    }

    try:
        if verbose:
            typer.echo("📡 Making request...")

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        if verbose:
            typer.echo(f"✅ Request successful (Status: {response.status_code})")
            typer.echo("🔍 Parsing HTML...")

        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract document content
        document_div = soup.find("div", class_="main-content")
        reference_div = soup.find("div", class_="bk_tt")

        results_found = False

        if document_div:
            if verbose:
                typer.echo("📄 Found document content:")
            typer.echo("=" * 80)
            typer.echo("DOCUMENT CONTENT:")
            typer.echo("=" * 80)
            typer.echo(document_div.get_text(" ").strip())
            typer.echo()
            results_found = True
        else:
            typer.echo(
                "❌ Error: The div with class 'document' was not found.", err=True
            )

        if reference_div:
            if verbose:
                typer.echo("📖 Found reference title:")
            typer.echo("=" * 80)
            typer.echo("REFERENCE TITLE:")
            typer.echo("=" * 80)
            typer.echo(reference_div.get_text(" ").strip())
            typer.echo()
            results_found = True
        else:
            typer.echo("❌ Error: The div with class 'bk_tt' was not found.", err=True)

        typer.echo("=" * 80)
        typer.echo("CITATION INFO:")
        typer.echo("=" * 80)
        typer.echo(f"Access online at: {url}")
        typer.echo()

        if not results_found:
            typer.echo(
                "⚠️  No expected content elements were found on this page.", err=True
            )
            raise typer.Exit(1)

        if verbose:
            typer.echo("✨ Scraping completed successfully!")

    except requests.exceptions.Timeout:
        typer.echo(f"⏰ Error: Request timed out after {timeout} seconds", err=True)
        raise typer.Exit(1)
    except requests.exceptions.ConnectionError:
        typer.echo(
            "🔌 Error: Failed to connect to the URL. Check your internet connection.",
            err=True,
        )
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        typer.echo(f"🚫 HTTP Error: {e}", err=True)
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        typer.echo(f"🔴 Request Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"💥 Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
