import pdfplumber
import re
from fuzzywuzzy import fuzz
import os
import unicodedata
import mimetypes

class data:
    def __init__(self,file_path):
        self.pdf_path = file_path
        self._vietnamese_pattern = r'[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]'
        self._text = None
        self._REMOVE_PHRASES = ["cộng hòa xã hội chủ nghĩa việt nam","độc lập - tự do - hạnh phúc"]
        self._THRESHOLD = 90

    def __read_pdf(self):
        self._text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    self._text += page_text + " "

    def __check_file_path(self):
        mime_type, _ = mimetypes.guess_type(self.pdf_path)
        if not os.path.isfile(self.pdf_path) or not self.pdf_path.lower().endswith('.pdf') or mime_type != 'application/pdf':
            return False
        return True
    
    def __hasvietnamese(self):
        return bool(re.search(self._vietnamese_pattern, self._text))
        
    def __normalize_text(self,text):
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __remove_similar_phrases(self):
        lines = self._text.split('\n')
        result = []
        for line in lines:
            original_line = line
            normalized_line = self.__normalize_text(line).lower()
            for phrase in self._REMOVE_PHRASES:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                matches = pattern.finditer(normalized_line)
                for match in matches:
                    matched_text = match.group()
                    if fuzz.partial_ratio(matched_text.lower(), phrase) >= self._THRESHOLD:
                        original_line = re.sub(re.escape(matched_text), '', original_line, flags=re.IGNORECASE)
            result.append(original_line.strip())
        self._text = '\n'.join(result)

    def __clean_text(self):
        self._text = unicodedata.normalize('NFC', self._text)
        self._text = self._text.replace('\r', '')
        self.__remove_similar_phrases()
        self._text = re.sub(r'\s+', ' ', self._text).strip()

    def load_pdf(self):
        if not self.__check_file_path():
            return False,False
        else:
            self.__read_pdf()
            if not self.__hasvietnamese():
                return True, False
            else:
                self.__clean_text()
                return True, True
    def read_text(self):
        return self._text