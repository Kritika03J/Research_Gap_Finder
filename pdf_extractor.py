
import uuid
from pathlib import Path

import fitz
import pdfplumber

from layout_pdf_extractor import (
    process_page,
    MIN_CHARS_PER_PAGE,
    OCR_DPI,
    FIGURE_DPI
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "extracted_papers"


def extract_uploaded_papers(uploaded_files):

    papers = {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:

        paper_name = uploaded_file.name

        # Create a separate folder for every uploaded PDF
        run_id = uuid.uuid4().hex[:8]

        paper_dir = OUTPUT_DIR / f"{Path(paper_name).stem}_{run_id}"

        paper_dir.mkdir(parents=True, exist_ok=True)

        tables_dir = paper_dir / "tables"
        figures_dir = paper_dir / "figures"

        tables_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(exist_ok=True)

        # Save uploaded PDF temporarily
        pdf_path = paper_dir / paper_name

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        extracted_pages = []

        doc = fitz.open(str(pdf_path))

        try:
            plpdf = pdfplumber.open(str(pdf_path))
        except Exception:
            plpdf = None

        try:

            for page_index in range(doc.page_count):

                page = doc[page_index]

                plpage = (
                    plpdf.pages[page_index]
                    if plpdf is not None
                    else None
                )

                try:

                    page_text = process_page(
                        page=page,
                        plpage=plpage,
                        page_index=page_index,
                        tables_dir=str(tables_dir),
                        figures_dir=str(figures_dir),
                        rel_base=str(paper_dir),
                        min_chars=MIN_CHARS_PER_PAGE,
                        ocr_dpi=OCR_DPI,
                        figure_dpi=FIGURE_DPI,
                        debug_dir=None
                    )

                    if page_text and page_text.strip():

                        extracted_pages.append({
                            "page": page_index + 1,
                            "text": page_text.strip()
                        })

                except Exception as e:

                    print(
                        f"Error extracting {paper_name}, "
                        f"page {page_index + 1}: {e}"
                    )

                finally:

                    if plpage is not None:
                        try:
                            plpage.close()
                        except Exception:
                            pass

        finally:

            doc.close()

            if plpdf is not None:
                plpdf.close()

        if extracted_pages:
            papers[paper_name] = extracted_pages

    return papers
