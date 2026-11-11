"""A stand-in for the model so tests never touch the network."""

import json
import re

from src.llm import LLMResult


class FakeLLM:
    """Answers PR-01 by citing the first source it was shown, and PR-02 with a fixed note."""

    def __init__(self, reply=None):
        self.reply, self.calls, self.cache_hits, self.prompts = reply, 0, 0, []

    def complete(self, system, user, max_tokens=700):
        self.calls += 1
        self.prompts.append((system, user))
        if self.reply is not None:
            text = self.reply(system, user) if callable(self.reply) else self.reply
        elif "check whether a support reply" in system:
            text = json.dumps({"claims": [{"text": "a claim", "supported": True, "source": "DOC-DATA-001"}]})
        elif "hand-over notes" in system:
            text = json.dumps(
                {
                    "summary": "Customer asks something.",
                    "customer_wants": "An answer.",
                    "relevant_docs": [],
                    "unsure_about": "",
                    "suggested_next_step": "Read the ticket.",
                }
            )
        else:
            ids = re.findall(r"^\[(DOC-[A-Z0-9-]+)\]", user, re.M)
            text = json.dumps(
                {
                    "can_answer": bool(ids),
                    "reply": f"Follow the documented steps [{ids[0]}]." if ids else "",
                    "citations": ids[:1],
                    "unsure_about": "",
                }
            )
        return LLMResult(text, "fake-model", "fake-model-1", attempts=1)
