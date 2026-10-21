# rulebook-ai — NBA Rulebook Assistant with RAG, MCP and Evaluation
Tech: Python, Ollama, Gemma, embeddings, RAG, MCP, pytest, GitHub Actions
- Built a question-answering assistant over the 2026-27 NBA playing rules with hybrid retrieval, section-cited answers and an MCP server (search, section lookup) usable from Claude Code and other MCP clients.
- Evaluated 6 strategies on 50 gold questions: grounding raised accuracy from 29% (model alone) to 95% with correct citations, eliminated invented citations, and declined all 8 out-of-scope questions.
- Showed a fixed retrieval pipeline beat tool-calling agents (71-91%) and embeddings beat keyword search (95% vs 76% recall@3); found a JSON-escaping bug hiding fractions from the agent.
