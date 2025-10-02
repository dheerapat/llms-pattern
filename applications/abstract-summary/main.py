import typer
from rich import print
from typing_extensions import Annotated
from pubmed import get_abstract, parse_pubmed_xml, search_journal
from llm import generate_answer, generate_keyword, classify_rct
from parsing_json import classify, get_keyword


def main(
    query: Annotated[str, typer.Argument()],
    strict: Annotated[
        bool,
        typer.Option(
            help="when True, only high quality abstract will be used to answer the question"
        ),
    ] = False,
):
    keyword = get_keyword(query)
    print(f"searching pubmed database with following keyword: '{keyword.keywords}'")

    journals = search_journal(keyword=keyword.keywords)
    ids = ", ".join(journals.esearchresult.idlist)

    abs = get_abstract(ids, "xml")
    parsed = parse_pubmed_xml(abs)
    context = []

    if strict:
        print("determine each abstract if it is high quality or not, stand by ...")
        for abs in parsed:
            abstract = abs.abstract.full_abstract.strip()
            if abstract:
                result = classify(abstract)
                if (
                    result.route == "randomized_controlled_trial"
                    or result.route == "meta_analysis"
                ):
                    context.append(abs)
            else:
                continue
    else:
        print("fetched all related abstract for you")
        context = parsed

    if len(context) == 0:
        print("no related abstract found, abort")
        return

    print(
        f"found {len(context)} most updated abstract related to the query, generating answer ...\n"
    )
    result = generate_answer(context, query)
    print("Answer:\n")
    print(result)


if __name__ == "__main__":
    typer.run(main)
