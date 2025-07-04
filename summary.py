import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from underthesea import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Summary:
    def __init__(self,
                 model_chunk_summary_path=r'models/chunk_summarizer',
                 model_summary_combiner=r'models/summary_combiner',
                 tokenizer_path=r'models/chunk_summarizer'):
        self._summary = None
        self._model_chunk_summary_path = model_chunk_summary_path
        self._model_summary_combiner_path = model_summary_combiner
        self._tokenizer_path = tokenizer_path
        self._chunk_summary = None
        self._summary_combiner = None
        self._tokenizer = None
        self._max_token_length = 512
        self._preferred_stop_chars = [".", "\n", "\n\n"]
        self._max_summary_length = 512
        self.__load_model()

    def __load_model(self):
        self._chunk_summarizer = AutoModelForSeq2SeqLM.from_pretrained(self._model_chunk_summary_path)
        self._summary_combiner = AutoModelForSeq2SeqLM.from_pretrained(self._model_summary_combiner_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        self._chunk_summarizer.to(device)
        self._summary_combiner.to(device)

    def __chunk_text(self, text):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0
        hard_limit = self._max_token_length
        soft_limit = int(self._max_token_length * 0.95)

        for sent in sentences:
            sent_len = len(self._tokenizer.encode(sent, add_special_tokens=False))
            if current_len + sent_len <= hard_limit:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                combined = " ".join(current_chunk).strip()
                if any(combined.endswith(char) for char in self._preferred_stop_chars) or current_len >= soft_limit:
                    chunks.append(combined)
                    current_chunk = [sent]
                    current_len = sent_len
                else:
                    temp = " ".join(current_chunk + [sent])
                    for stop_char in self._preferred_stop_chars:
                        if stop_char in temp:
                            parts = temp.split(stop_char)
                            first_part = stop_char.join(parts[:-1]) + stop_char
                            rest = parts[-1]
                            chunks.append(first_part.strip())
                            current_chunk = [rest.strip()]
                            current_len = len(self._tokenizer.encode(rest, add_special_tokens=False))
                            break
                    else:
                        chunks.append(" ".join(current_chunk).strip())
                        current_chunk = [sent]
                        current_len = sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return [chunk for chunk in chunks if chunk]

    def __summarize_hierarchical(self, text):
        chunks = self.__chunk_text(text)
        if not chunks:
            return "Không thể phân đoạn văn bản để tóm tắt."
        chunk_summaries = []

        self._chunk_summarizer.eval()
        self._summary_combiner.eval()

        with torch.no_grad():
            # Xác định batch_size dựa trên số chunk
            if torch.cuda.is_available():
                if len(chunks) <= 2:
                    batch_size = 2
                elif len(chunks) <= 7:
                    batch_size = 4
                else:
                    batch_size = 8

                # Tóm tắt từng chunk theo batch
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    inputs = self._tokenizer(
                        batch_chunks,
                        max_length=self._max_token_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        return_token_type_ids=False
                    )
                    inputs = {k: v.to(self._chunk_summarizer.device) for k, v in inputs.items()}
                    summary_ids = self._chunk_summarizer.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=self._max_summary_length,
                        num_beams=4,
                        length_penalty=1.0
                    )
                    summaries = [self._tokenizer.decode(summary_id, skip_special_tokens=True) for summary_id in summary_ids]
                    chunk_summaries.extend(summaries)
            else:
                # Xử lý từng chunk khi không có CUDA
                for chunk in chunks:
                    inputs = self._tokenizer(
                        chunk,
                        max_length=self._max_token_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                        return_token_type_ids=False
                    )
                    inputs = {k: v.to(self._chunk_summarizer.device) for k, v in inputs.items()}
                    summary_ids = self._chunk_summarizer.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=self._max_summary_length,
                        num_beams=4,
                        length_penalty=1.0
                    )
                    summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    chunk_summaries.append(summary)

            combined_summary = " ".join(chunk_summaries)

            # Tổng hợp tóm tắt nếu cần
            while len(self._tokenizer.encode(combined_summary, add_special_tokens=False)) > self._max_token_length:
                chunks = self.__chunk_text(combined_summary)
                chunk_summaries = []

                if torch.cuda.is_available():
                    # Xác định batch_size cho summary_combiner
                    if len(chunks) <= 2:
                        batch_size = 2
                    elif len(chunks) <= 7:
                        batch_size = 4
                    else:
                        batch_size = 8

                    # Tóm tắt theo batch
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        inputs = self._tokenizer(
                            batch_chunks,
                            max_length=self._max_token_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                            return_token_type_ids=False
                        )
                        inputs = {k: v.to(self._summary_combiner.device) for k, v in inputs.items()}
                        summary_ids = self._summary_combiner.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=self._max_summary_length,
                            num_beams=4,
                            length_penalty=1.0
                        )
                        summaries = [self._tokenizer.decode(summary_id, skip_special_tokens=True) for summary_id in summary_ids]
                        chunk_summaries.extend(summaries)
                else:
                    # Xử lý từng chunk khi không có CUDA
                    for chunk in chunks:
                        inputs = self._tokenizer(
                            chunk,
                            max_length=self._max_token_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                            return_token_type_ids=False
                        )
                        inputs = {k: v.to(self._summary_combiner.device) for k, v in inputs.items()}
                        summary_ids = self._summary_combiner.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_length=self._max_summary_length,
                            num_beams=4,
                            length_penalty=1.0
                        )
                        summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        chunk_summaries.append(summary)

                combined_summary = " ".join(chunk_summaries)

        return combined_summary.strip()

    def summary_content(self, text):
        summary = self.__summarize_hierarchical(text)
        return summary