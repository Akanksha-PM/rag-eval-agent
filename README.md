# RAG Eval Agent

A general-purpose documentation comparison tool. It ingests documentation for
any set of competing software products — registered dynamically at runtime,
never hardcoded — and uses a retrieval-augmented, LLM-judged evaluation agent
to answer questions and compare products against each other. New products can
be added or removed from the registry at any time without changing any code,
so the same agent works for whatever set of products you point it at.
