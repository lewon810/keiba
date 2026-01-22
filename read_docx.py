from docx import Document
import sys

def read_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading file: {e}"

if __name__ == "__main__":
    file_path = "docs/競馬予測論文の探索と解説.docx"
    content = read_docx(file_path)
    output_file = "docs_content.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(content))
    print(f"Saved content to {output_file}")
