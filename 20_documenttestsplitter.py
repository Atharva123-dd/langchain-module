from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# for python code

text="""
class RecursiveCharacterTextSplitter:
    
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        separators=None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators
        self.separators = separators or [
            "\\n\\n",   # paragraph
            "\\n",      # line
            ". ",       # sentence
            " ",        # word
            ""          # character fallback
        ]

    def split_text(self, text):

        chunks = []

        # Start recursive splitting
        self._split_recursive(text, self.separators, chunks)

        return chunks

    def _split_recursive(self, text, separators, chunks):

        # Base case
        if len(text) <= self.chunk_size:
            chunks.append(text.strip())
            return

        # No separators left
        if not separators:
            for i in range(0, len(text), self.chunk_size):
                chunks.append(text[i:i+self.chunk_size])
            return

        separator = separators[0]

        # Character-level split fallback
        if separator == "":
            for i in range(0, len(text), self.chunk_size):
                chunks.append(text[i:i+self.chunk_size])
            return

        parts = text.split(separator)

        current_chunk = ""

        for part in parts:

            # Rebuild piece with separator
            if current_chunk:
                temp_chunk = current_chunk + separator + part
            else:
                temp_chunk = part

            # If chunk fits
            if len(temp_chunk) <= self.chunk_size:
                current_chunk = temp_chunk

            else:

                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Overlap handling
                overlap = current_chunk[-self.chunk_overlap:] if current_chunk else ""

                current_chunk = overlap + separator + part

                # If still too large -> recurse deeper
                if len(current_chunk) > self.chunk_size:
                    self._split_recursive(
                        current_chunk,
                        separators[1:],
                        chunks
                    )
                    current_chunk = ""

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=20,
    
)

spitted_text = splitter.split_text(text)

for i, chunk in enumerate(spitted_text):
    print(f"Chunk {i}: {chunk}")
