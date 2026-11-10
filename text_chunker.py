
def create_chunks(
    pages,
    chunk_size=4000,
    overlap=500
):

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks
