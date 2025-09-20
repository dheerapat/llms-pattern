from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    title: str
    section_path: str
    content: str


def chunk_markdown(md_text: str, doc_id: str) -> list[Chunk]:
    lines = md_text.strip().splitlines()
    if not lines:
        return []

    chunks = []
    current_title = ""
    section_stack = []
    content_buffer = []

    def save_chunk():
        if not content_buffer or not section_stack:
            return

        chunk = Chunk(
            doc_id=doc_id,
            title=current_title,
            section_path=" > ".join(s["text"] for s in section_stack),
            content="\n".join(content_buffer).strip(),
        )
        chunks.append(chunk)
        content_buffer.clear()

    for line in lines:
        if line.strip() == "---":
            break

        stripped_line = line.lstrip()
        if stripped_line.startswith("#"):
            level = 0
            while level < len(stripped_line) and stripped_line[level] == "#":
                level += 1

            if (
                1 <= level <= 6
                and len(stripped_line) > level
                and stripped_line[level] == " "
            ):
                heading_text = stripped_line[level + 1 :].strip()
                save_chunk()

                if level == 1:
                    current_title = heading_text
                    section_stack.clear()
                else:
                    while section_stack and section_stack[-1]["level"] >= level:
                        section_stack.pop()
                    section_stack.append({"level": level, "text": heading_text})
                continue

        content_buffer.append(line)

    save_chunk()

    return chunks
