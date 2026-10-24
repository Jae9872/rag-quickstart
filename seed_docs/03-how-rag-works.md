# How RAG works (a short note)

RAG stands for Retrieval-Augmented Generation. It is a pattern for making
a language model answer from your own documents instead of only from what
it memorised during training.

## Step 1: Embeddings

An embedding turns text into a list of numbers (a vector) that captures
meaning. Texts with similar meaning get vectors that point in similar
directions. A small model called all-MiniLM-L6-v2 turns sentences into
384-number vectors, runs locally, and needs no API key.

## Step 2: Retrieval

Documents are split into chunks, embedded, and stored in a vector database
such as ChromaDB. When a question arrives, it is embedded the same way,
and the database returns the chunks whose vectors are closest to the
question vector. This is similarity search, not keyword search: it finds
meaning, not just matching words.

## Step 3: Generation

The retrieved chunks are pasted into the language model's prompt along with
the question, and the model writes an answer grounded in those chunks. The
answer should cite which chunks it used, so a human can check it. If no
chunk is relevant, the honest answer is "I don't know from these documents"
— not a confident guess.

## Step 4: Evals

Evals are tests for a RAG system. Good eval questions include: does the
answer cite the right chunk? Does it refuse when the documents do not
contain the answer? Does it stay accurate when new documents are added?
A RAG system without evals is a demo. A RAG system with evals is a product.

## Why RAG matters for regulated work

In insurance and other regulated industries, an answer without a source is
not usable. RAG gives every answer a paper trail: the exact chunk it came
from. Combined with a human approval gate before any real decision or
payment, RAG becomes a tool a compliance officer can sign off on.
