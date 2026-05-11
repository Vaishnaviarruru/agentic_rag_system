import os
import csv
import PyPDF2
from src.config import DATA_DIR


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    text = ""
    with open(filepath, "rb") as f:
        try:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"  PDF read error in {filepath}: {e}")
    return text


def load_csv(filepath: str) -> str:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
            rows.append(row_text)
    return "\n".join(rows)


def load_from_folder(folder: str = DATA_DIR) -> list[dict]:
    """Load all supported documents from a folder on disk."""
    documents = []

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return documents

    for filename in sorted(os.listdir(folder)):
        if filename.startswith("."):
            continue

        filepath = os.path.join(folder, filename)
        ext = filename.lower().rsplit(".", 1)[-1]

        try:
            if ext == "txt":
                content = load_txt(filepath)
            elif ext == "pdf":
                content = load_pdf(filepath)
            elif ext == "csv":
                content = load_csv(filepath)
            else:
                continue

            if content.strip():
                documents.append({"source": filename, "content": content})
                print(f"  Loaded: {filename} ({len(content):,} chars)")
            else:
                print(f"  Warning: {filename} appears empty — skipped")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    return documents


def load_from_bytes(filename: str, file_bytes: bytes) -> dict | None:
    """Load a document from raw bytes (used by Streamlit uploader)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    content = ""

    try:
        if ext == "txt":
            content = file_bytes.decode("utf-8", errors="ignore")

        elif ext == "pdf":
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    content += extracted + "\n"

        elif ext == "csv":
            import io
            text = file_bytes.decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            rows = []
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            content = "\n".join(rows)

    except Exception as e:
        print(f"  Error parsing {filename}: {e}")

    if content.strip():
        return {"source": filename, "content": content}
    return None


def load_all_documents(folder: str = DATA_DIR) -> list[dict]:
    print(f"Loading documents from '{folder}'...")
    docs = load_from_folder(folder)
    print(f"Total documents loaded: {len(docs)}")
    return docs