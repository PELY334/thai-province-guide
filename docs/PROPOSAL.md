# Proposal (week 1)

**Thai Province Guide.** Students, teachers and travellers who ask a chatbot about a Thai
province (its seal, provincial tree, rivers, parks, history) get confident answers from a
small model that are often invented. I will build a question-answering app over the English
Wikipedia articles for all 77 provinces that answers only from those articles, cites the
passage, and abstains when the articles are silent; questions that need two facts and
arithmetic ("how much larger is Chiang Mai than Kanchanaburi?") go to a tool-calling agent
with a search tool and a safe calculator. Techniques: structured JSON output validated with
Pydantic, RAG with citations and a `min_score` floor, an agent loop with allow-listed tools
and a step limit, and a FastAPI + SSE interface, all against `gemma-4-E4B-it` on the class
proxy. Evaluation: a hand-written test set of ~40 questions (single-fact, reverse lookups,
multi-hop arithmetic, unanswerable) with gold answers and gold evidence passages; metrics
recall@5 / MRR for retrieval and answer accuracy, abstention and citation precision end to
end; baselines: the same model with no retrieval, BM25-only retrieval, and default
250-word chunks with no floor. A separate dev set is used for tuning.
