# The Go Gopher

The Go gopher is the mascot of the Go programming language. It was designed by
Renée French, the same artist who created the mascot for the Plan 9 operating
system. The gopher's design is deliberately simple so it can be drawn and
remixed easily by the community. Its colours are typically light blue, matching
the Go brand.

# Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, is a technique for making language
models answer questions using specific documents rather than only their trained
knowledge. The pipeline has three stages. First, documents are split into
chunks and converted into vectors using an embedding model. Second, when a user
asks a question, the question is embedded and compared against the stored
vectors to retrieve the most relevant chunks. Third, those chunks are inserted
into the prompt so the model's answer is grounded in real source material. RAG
reduces hallucination and lets a model use private or up-to-date information it
was never trained on.

# Cosine Similarity

Cosine similarity measures how similar two vectors are by looking at the angle
between them rather than their length. A value of 1.0 means the vectors point in
exactly the same direction, 0.0 means they are unrelated, and -1.0 means they
point in opposite directions. It is the workhorse metric of vector search
because it ignores magnitude and focuses purely on direction, which is what
captures meaning in an embedding space.
