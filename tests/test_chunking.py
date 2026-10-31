from maint_copilot.chunking import chunk_page, split_blocks


def words(text: str) -> int:
    return len(text.split())


PAGE = """# 2.1 FUEL SERVICING

WARNING Ground the aircraft before servicing.

Connect the bonding cable. Check the deadman control.

| Item | Torque |
|---|---|
| Flange bolt | 35 in-lb |
| Cap | 10 in-lb |

## 2.2 DEFUELING
Stop all operations if a leak develops."""


def test_blocks_follow_headings_paragraphs_and_tables():
    blocks = split_blocks(PAGE)
    assert blocks[0] == "# 2.1 FUEL SERVICING"
    assert any(b.startswith("| Item") and b.count("\n") == 3 for b in blocks)
    assert blocks[-1].startswith("## 2.2 DEFUELING")


def test_chunks_respect_budget_and_keep_all_words():
    chunks = chunk_page(7, PAGE, max_tokens=12, length=words)
    assert all(words(c.text) <= 12 for c in chunks)
    assert sum(words(c.text) for c in chunks) == words(PAGE)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page_id for c in chunks} == {7}


def test_oversized_paragraph_is_split_at_sentences():
    long_para = " ".join(f"Sentence number {i} ends here." for i in range(20))
    chunks = chunk_page(1, long_para, max_tokens=15, length=words)
    assert len(chunks) > 1
    assert all(words(c.text) <= 15 for c in chunks)
    assert all(c.text.endswith("here.") for c in chunks)


def test_empty_page_has_no_chunks():
    assert chunk_page(3, "   \n\n", max_tokens=50, length=words) == []
