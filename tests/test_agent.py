"""Agent loop tests with a scripted fake model (no API calls)."""

import copy
from types import SimpleNamespace

import pandas as pd

from maint_copilot.agent import Budget, Library, run_agent

CORPUS = pd.DataFrame(
    {
        "corpus_id": [1, 2, 3, 4],
        "doc_id": ["TO_A", "TO_A", "TO_A", "TO_B"],
        "page_number_in_doc": [1, 2, 3, 1],
        "markdown": [
            "Intro.",
            "Torque flange bolts to 35 in-lb.",
            "Continued: check seal.",
            "Other.",
        ],
    }
)


class Block(SimpleNamespace):
    def model_dump(self):
        return {k: v for k, v in self.__dict__.items()}


def tool(name, **inp):
    return Block(type="tool_use", id=f"id_{name}_{len(str(inp))}", name=name, input=inp)


class FakeClient:
    """Returns scripted responses in order and records the requests it received."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(copy.deepcopy(kw))
        content = self.script.pop(0)
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        return SimpleNamespace(content=content, usage=usage, stop_reason="tool_use")


def lib():
    return Library(CORPUS, search=lambda q, k: [2, 4][:k])


def test_search_read_submit_happy_path():
    client = FakeClient(
        [
            [tool("search_pages", query="flange torque")],
            [tool("read_pages", page_ids=["P2"])],
            [
                tool(
                    "submit_answer",
                    answer="35 in-lb",
                    citations=["P2"],
                    insufficient_evidence=False,
                )
            ],
        ]
    )
    out = run_agent(client, "m", "What torque?", lib())
    assert out["answer"] == "35 in-lb" and out["citations"] == [2]
    assert out["tool_calls"] == 2 and out["stop_cause"] == "submitted"
    assert out["context_page_ids"] == [2] and out["pages_seen_in_search"] == [2, 4]
    assert [s["tool"] for s in out["trace"]] == ["search_pages", "read_pages", "submit_answer"]


def test_citing_an_unread_page_is_rejected_then_corrected():
    client = FakeClient(
        [
            [tool("search_pages", query="torque")],
            [tool("submit_answer", answer="35", citations=["P2"], insufficient_evidence=False)],
            [tool("read_pages", page_ids=["P2"])],
            [
                tool(
                    "submit_answer",
                    answer="35 in-lb",
                    citations=["P2"],
                    insufficient_evidence=False,
                )
            ],
        ]
    )
    out = run_agent(client, "m", "q", lib())
    assert out["answer"] == "35 in-lb" and out["citations"] == [2]
    assert any(s.get("result") == "rejected: unread citations" for s in out["trace"])
    rejected = client.requests[2]["messages"][-1]["content"][0]
    assert rejected["is_error"] and "not read in full" in rejected["content"]


def test_budget_forces_submission():
    client = FakeClient(
        [
            [tool("search_pages", query="a")],
            [tool("search_pages", query="b")],
            [
                tool(
                    "submit_answer", answer="partial", citations=["P9"], insufficient_evidence=False
                )
            ],
        ]
    )
    out = run_agent(client, "m", "q", lib(), budget=Budget(max_tool_calls=2))
    assert client.requests[2]["tool_choice"] == {"type": "tool", "name": "submit_answer"}
    assert out["stop_cause"] == "budget"
    assert out["citations"] == [] and out["dropped_citations"] == [9]


def test_read_adjacent_stays_in_the_same_manual():
    client = FakeClient(
        [
            [tool("read_adjacent", page_id="P2", before=1, after=2)],
            [tool("submit_answer", answer="x", citations=["P3"], insufficient_evidence=False)],
        ]
    )
    out = run_agent(client, "m", "q", lib())
    assert out["context_page_ids"] == [1, 3]


def test_unknown_page_id_returns_an_error_to_the_model():
    client = FakeClient(
        [
            [tool("read_pages", page_ids=["P999"])],
            [tool("submit_answer", answer="none", citations=[], insufficient_evidence=True)],
        ]
    )
    out = run_agent(client, "m", "q", lib())
    first_result = client.requests[1]["messages"][-1]["content"][0]["content"]
    assert first_result.startswith("Error") and out["insufficient_evidence"]


def test_empty_submission_is_returned_as_an_error():
    client = FakeClient(
        [
            [tool("read_pages", page_ids=["P2"])],
            [tool("submit_answer")],
            [
                tool(
                    "submit_answer",
                    answer="35 in-lb",
                    citations=["P2"],
                    insufficient_evidence=False,
                )
            ],
        ]
    )
    out = run_agent(client, "m", "q", lib())
    assert out["answer"] == "35 in-lb"
    assert any(s.get("result") == "rejected: empty answer" for s in out["trace"])
    assert client.requests[2]["tool_choice"] == {"type": "tool", "name": "submit_answer"}


def test_reading_more_than_five_pages_says_so():
    client = FakeClient(
        [
            [tool("read_pages", page_ids=["P1", "P2", "P3", "P4", "P1", "P2"])],
            [tool("submit_answer", answer="x", citations=["P1"], insufficient_evidence=False)],
        ]
    )
    run_agent(client, "m", "q", lib())
    result = client.requests[1]["messages"][-1]["content"][0]["content"]
    assert "Only the first 5 of 6" in result


class FakeParseClient(FakeClient):
    """Adds messages.parse: returns scripted parsed outputs for the self-check call."""

    def __init__(self, script, parsed):
        super().__init__(script)
        self.parsed = list(parsed)

    def parse(self, **kw):
        out = self.parsed.pop(0)
        usage = SimpleNamespace(
            input_tokens=50,
            output_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(parsed_output=out, usage=usage, stop_reason="end_turn")


def checked(revised, citations, claims):
    return SimpleNamespace(
        revised_answer=revised,
        citations=citations,
        claims=[SimpleNamespace(claim=c, supported=s, page_id=p, quote=q) for c, s, p, q in claims],
    )


def test_self_check_replaces_draft_and_keeps_it_in_the_record():
    client = FakeParseClient(
        [
            [tool("read_pages", page_ids=["P2"])],
            [
                tool(
                    "submit_answer",
                    answer="35 in-lb; also 50 in-lb",
                    citations=["P2"],
                    insufficient_evidence=False,
                )
            ],
        ],
        [
            checked(
                "35 in-lb",
                ["P2", "P4"],
                [
                    ("Torque is 35 in-lb", True, "P2", "Torque flange bolts to 35 in-lb"),
                    ("Also 50 in-lb", False, None, ""),
                ],
            )
        ],
    )
    out = run_agent(client, "m", "What torque?", lib(), self_check=True)
    assert out["answer"] == "35 in-lb" and out["draft_answer"] == "35 in-lb; also 50 in-lb"
    assert out["citations"] == [2]  # P4 was never read, so it cannot be cited
    sc = out["self_check"]
    assert sc["applied"] and sc["claims_removed"] == 1 and sc["quote_verified_share"] == 1.0
    assert out["trace"][-1]["tool"] == "self_check"
    assert out["usage"]["input_tokens"] == 250  # two agent turns + self-check


def test_self_check_skipped_when_agent_declines():
    client = FakeParseClient(
        [[tool("submit_answer", answer="Not found", citations=[], insufficient_evidence=True)]],
        [],
    )
    out = run_agent(client, "m", "q", lib(), self_check=True)
    assert out["self_check"] is None and out["answer"] == "Not found"


def test_failed_self_check_keeps_the_draft():
    client = FakeParseClient(
        [
            [tool("read_pages", page_ids=["P2"])],
            [
                tool(
                    "submit_answer",
                    answer="35 in-lb",
                    citations=["P2"],
                    insufficient_evidence=False,
                )
            ],
        ],
        [None, None],
    )
    out = run_agent(client, "m", "q", lib(), self_check=True)
    assert out["answer"] == "35 in-lb" and not out["self_check"]["applied"]
