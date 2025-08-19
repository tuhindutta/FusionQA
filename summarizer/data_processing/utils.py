from pypdf import PdfReader
import re


def clean_text_regex(text):
    pattern = r"[^a-zA-Z0-9\s" + r"]"
    cleaned_text = '\n'.join([i.strip() for i in re.sub(pattern, "", text).split('\n')]).split('\n')
    cleaned_text = '\n'.join([re.sub(r'\s+', ' ', i) for i in cleaned_text])
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    return cleaned_text


class Pdf:

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        reader = PdfReader(pdf_path)
        pagewise_text = []
        for page in reader.pages:
            pagewise_text.append(page.extract_text())
        return pagewise_text

    def read(self, path):
        extracted_content = self.extract_text_from_pdf(path)
        processed_text = [clean_text_regex(i) for i in extracted_content]
        processed_text_blob = '||PAGE_BREAK||'.join(processed_text)
        return processed_text_blob