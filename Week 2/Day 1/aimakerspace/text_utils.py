import os
from typing import List
import tiktoken

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents

class CharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        assert chunk_size > chunk_overlap, "Chunk size must be greater than chunk overlap"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i: i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

class TokenTextSplitter:
    def __init__(self, max_tokens=2048, tokenizer_name="cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_tokens = max_tokens

    def recursive_split(self, doc):
        if len(self.tokenizer.encode(doc)) <= self.max_tokens:
            return [doc]

        paragraphs = doc.split('\n\n')
        if len(paragraphs) == 1:
            sentences = doc.split('. ')
            half = len(sentences) // 2
            return self.recursive_split('. '.join(sentences[:half])) + \
                   self.recursive_split('. '.join(sentences[half:]))
        
        half = len(paragraphs) // 2
        return self.recursive_split('\n\n'.join(paragraphs[:half])) + \
               self.recursive_split('\n\n'.join(paragraphs[half:]))

    def split(self, text: str) -> List[str]:
        return self.recursive_split(text)

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
