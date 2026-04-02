def test_chunk_fixed_overlap():
    text = "abcdefghij"
    chunks = chunk_fixed(text, chunk_size=4, overlap=1)
    assert chunks == ["abcd", "dghi", "ghij"]  # Example expected


def test_recursive_split():
    text = "Para1\n\nPara2\n\nPara3"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=0, separators=["\n\n"]
    )
    chunks = splitter.split_text(text)
    assert chunks == ["Para1", "Para2", "Para3"]
