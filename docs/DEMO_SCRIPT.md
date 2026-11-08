# Demo video script (about 5 minutes)

Record the screen with OBS or the Windows Game Bar (Win + Alt + R). Start the app first:
`.venv\Scripts\activate` then `uvicorn app.server:app`, open http://127.0.0.1:8000.

| Time | Show | Say (EN; Thai notes in brackets) |
|---|---|---|
| 0:00–0:30 | README on GitHub | "Thai Province Guide answers questions about Thailand's 77 provinces only from their Wikipedia articles, cites the passage, and abstains when the articles don't say. It runs on gemma-4-E4B-it through the class proxy." [แอปตอบคำถามเรื่อง 77 จังหวัด จาก Wikipedia เท่านั้น มีอ้างอิง และงดตอบถ้าไม่มีข้อมูล] |
| 0:30–1:00 | `docs/architecture.png` | "Offline we chunk the articles into 60-word, section-aware chunks and embed them locally. Online, FastAPI sends a question to single-shot RAG or to an agent with two tools, search and a safe calculator. Every model answer is JSON checked by Pydantic, and citations are checked against the sources we actually sent." |
| 1:00–1:45 | UI, mode **RAG**: "What is the provincial tree of Lampang?" | **Success case.** Point at the answer (Indian Elm), the citation chip, the highlighted source, and the Wikipedia revision link. |
| 1:45–2:45 | UI, mode **Agent**: "How many square kilometres larger is Chiang Mai province than Kanchanaburi province?" | **Agent success.** Steps stream in: two `search_provinces` calls, then `calculator(22311 - 19482)`, answer 2,829 km² with two citations. "Single-shot RAG got this kind of question right only 1–3 times out of 4; the agent got 4 of 4." |
| 2:45–3:15 | UI: "What is the best pad thai restaurant in Phuket?" | **Correct abstention.** "Nothing clears the 0.72 similarity floor, so it says it can't answer without even calling the model." |
| 3:15–4:00 | UI, mode RAG: "Which dam in Tak province created the largest artificial lake in Thailand?" | **Honest failure (q29).** It says the sources don't name the dam. "The 60-word chunk about the lake doesn't contain the dam's name; it is in the previous chunk, which falls under the floor. Smaller chunks raised recall@5 from 0.897 to 0.971, but this is their cost." |
| (optional) | UI, mode Agent: "How much larger is Chiang Mai province than Lamphun province, in km²?" | **Second honest failure, found in manual testing.** The Lamphun article has no total area, only "total forest area is 2,588 km² or 57 percent". The agent subtracts the forest area and answers 19,723 km², with citations that look convincing. "Citations make this checkable: click source 6 and you can see it says *forest* area. Next step: make the agent check the unit/quantity it cites matches the question." |
| 4:00–4:45 | `eval/results/RESULTS.md` | "Without retrieval the model got 2 of 34 right. With RAG, 30 to 31. Recall@5 0.971 with 60-word chunks vs 0.897 with 250. I tuned min_score on a separate dev set, and on test it stops 4 of 6 unanswerable questions before any LLM call, at the cost of one answerable question." |
| 4:45–5:00 | terminal: `python -m pytest -q` | "22 offline tests. All numbers are reproducible from the eval scripts. Thanks." |

Tip: run each demo question once before recording so the embedding model is loaded.
If the proxy is slow on the day, the RAG examples usually return in 1–3 seconds.
