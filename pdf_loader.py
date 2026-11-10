
import fitz


def load_papers(uploaded_files):

    papers = {}

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        pdf = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        pages = []

        for page_number, page in enumerate(pdf):

            text = page.get_text("text")

            if text.strip():

                pages.append({
                    "page": page_number + 1,
                    "text": text.strip()
                })

        pdf.close()

        papers[uploaded_file.name] = pages

    return papers
