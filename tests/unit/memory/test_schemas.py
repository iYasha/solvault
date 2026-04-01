import pytest
from pydantic import ValidationError

from sol.memory.schemas import Confidence, ExtractionResult, MemoryFact, MemoryType


class TestMemoryFact:
    """MemoryFact should validate fields and provide sensible defaults."""

    def test_valid_fact(self):
        """Given all valid fields, when constructed, then all fields are set correctly."""
        fact = MemoryFact(content="User likes Python", type="user", confidence="confirmed", tags=["tech"])

        assert fact.content == "User likes Python", "Content should match"
        assert fact.type == MemoryType.USER, "Type should be user"
        assert fact.confidence == Confidence.CONFIRMED, "Confidence should be confirmed"
        assert fact.tags == ["tech"], "Tags should match"

    def test_default_type_is_facts(self):
        """Given no type specified, when constructed, then type defaults to facts."""
        fact = MemoryFact(content="Some fact")

        assert fact.type == MemoryType.FACTS, "Default type should be facts"

    def test_default_confidence_is_inferred(self):
        """Given no confidence specified, when constructed, then confidence defaults to inferred."""
        fact = MemoryFact(content="Some fact")

        assert fact.confidence == Confidence.INFERRED, "Default confidence should be inferred"

    def test_default_tags_is_empty_list(self):
        """Given no tags specified, when constructed, then tags defaults to empty list."""
        fact = MemoryFact(content="Some fact")

        assert fact.tags == [], "Default tags should be empty list"

    def test_rejects_invalid_type(self):
        """Given an invalid type value, when constructed, then ValidationError is raised."""
        with pytest.raises(ValidationError):
            MemoryFact(content="Some fact", type="invalid_type")

    def test_rejects_invalid_confidence(self):
        """Given an invalid confidence value, when constructed, then ValidationError is raised."""
        with pytest.raises(ValidationError):
            MemoryFact(content="Some fact", confidence="maybe")


class TestExtractionResult:
    """ExtractionResult should hold a list of MemoryFact objects."""

    def test_default_facts_is_empty_list(self):
        """Given no facts, when constructed, then facts defaults to empty list."""
        result = ExtractionResult()

        assert result.facts == [], "Default facts should be empty list"

    def test_accepts_list_of_memory_facts(self):
        """Given a list of MemoryFact objects, when constructed, then stored correctly."""
        facts = [MemoryFact(content="Fact 1"), MemoryFact(content="Fact 2")]
        result = ExtractionResult(facts=facts)

        assert len(result.facts) == 2, "Should contain two facts"
        assert result.facts[0].content == "Fact 1", "First fact content should match"
