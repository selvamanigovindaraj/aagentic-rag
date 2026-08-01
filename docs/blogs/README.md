# Engineering Agentic RAG: a three-part series

This series follows one question through the architecture of this repository:
how do private company documents become answers that are useful, authorized,
and grounded?

The progression matters:

1. [Building an Ingestion Core for Advanced RAG](01-building-the-ingestion-core.md)
   explains how a source file becomes searchable chunks, hierarchical
   summaries, and provenance-linked graph facts.
2. [Beyond Vector Search](02-retrieval-for-multihop-questions.md) explains how
   the query path selects and combines those representations, grades the
   resulting evidence, and repairs weak searches.
3. [From Advanced RAG to Agentic RAG](03-turning-rag-into-an-agentic-system.md)
   explains how LangGraph turns those operations into a bounded,
   checkpointed, self-correcting state machine.

These are implementation notes, not generic RAG tutorials. File names,
settings, schemas, branches, and limitations refer to the current repository.
Examples use fictional company data unless stated otherwise.

