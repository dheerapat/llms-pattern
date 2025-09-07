import requests
from bs4 import BeautifulSoup
import typer
from pydantic import BaseModel

app = typer.Typer(help="Web scraper for extracting content from NCBI and similar pages")


class ScrapeResult(BaseModel):
    document: str
    ref: str


@app.command()
def scrape(url: str = typer.Argument(..., help="URL to scrape")):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
    }

    doc = ""
    ref = ""

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        document_div = soup.find("div", class_="main-content")
        reference_div = soup.find("div", class_="bk_tt")

        results_found = False

        if document_div:
            doc = document_div.get_text(" ").strip()
            typer.echo("DOCUMENT CONTENT:")
            typer.echo(doc)
            typer.echo()
            results_found = True
        else:
            typer.echo("Error: The div with class 'document' was not found.", err=True)

        if reference_div:
            ref = reference_div.get_text(" ").strip()
            typer.echo("REFERENCE TITLE:")
            typer.echo(ref)
            typer.echo()
            results_found = True
        else:
            typer.echo("Error: The div with class 'bk_tt' was not found.", err=True)

        typer.echo("CITATION INFO:")
        typer.echo(f"Available from: {url}")
        typer.echo()

        if not results_found:
            typer.echo(
                "No expected content elements were found on this page.", err=True
            )
            raise typer.Exit(1)

        return ScrapeResult(document=doc, ref=ref)

    except requests.exceptions.Timeout:
        typer.echo(f"Error: Request timed out", err=True)
        raise typer.Exit(1)
    except requests.exceptions.ConnectionError:
        typer.echo(
            "Error: Failed to connect to the URL. Check your internet connection.",
            err=True,
        )
        raise typer.Exit(1)
    except requests.exceptions.HTTPError as e:
        typer.echo(f"HTTP Error: {e}", err=True)
        raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Request Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
