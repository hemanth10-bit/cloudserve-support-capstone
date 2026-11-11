# Attribution and AI-use declaration

## AI assistance
Substantial parts of this repository (source code, tests, documentation, workbook content) were written by
**Claude Code** (model `claude-sonnet-5-5`, Anthropic) in working sessions directed by the author, Hemanth Kumar P.
The author's review of that material: [AUTHOR TO COMPLETE: say what you reviewed, changed and ran yourself].
Each commit that includes AI-written material carries a `Co-Authored-By: Claude` trailer.

## Third-party components
| Component | Use | Licence |
|---|---|---|
| all-MiniLM-L6-v2 (sentence-transformers) | embeddings for retrieval and classification | Apache-2.0 |
| Chroma | vector store | Apache-2.0 |
| scikit-learn, NumPy, pandas, SciPy | classifier, calibration, metrics | BSD-3-Clause |
| SQLAlchemy | decision log | MIT |
| LangGraph / LangChain core | orchestration (when wired in) | MIT |
| OpenAI Python SDK | client for the OpenRouter API | Apache-2.0 |
| FastAPI, Uvicorn, Pydantic | API layer | MIT / BSD / MIT |
| pytest, flake8, black | tests and style | MIT |
| meta-llama/llama-3.1-8b-instruct via OpenRouter | answer drafting (planned, B-08) | Llama 3.1 Community Licence |

## Data
`data/*.json` (tickets, documentation, reference answers) were supplied with the course pack for the CloudServe
capstone; they are fictional.
