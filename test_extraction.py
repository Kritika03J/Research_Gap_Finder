
from pdf_extractor import extract_uploaded_papers


class TestPDF:

    def __init__(self, path):
        self.name = path.split("\\")[-1]

        with open(path, "rb") as f:
            self.data = f.read()

    def getvalue(self):
        return self.data


pdf_files = [
    TestPDF(r"C:/Users/karun/OneDrive/Desktop/dataset_pprs/10ovarian2.pdf"),
    TestPDF(r"C:/Users/karun/OneDrive/Desktop/dataset_pprs/01breast1.pdf")
]

papers = extract_uploaded_papers(pdf_files)

print("Number of papers:", len(papers))

for paper_name, pages in papers.items():

    print("\nPaper:", paper_name)
    print("Number of extracted pages:", len(pages))

    for page in pages[:2]:

        print("\nPage:", page["page"])
        print(page["text"][:500])
