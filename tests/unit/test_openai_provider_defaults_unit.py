from fractalsemantics.embeddings.openai_provider import OpenAIEmbeddingProvider


def test_openai_provider_uses_legacy_default_model_for_backwards_compatibility() -> None:
    provider = OpenAIEmbeddingProvider()
    assert provider.model == "text-embedding-ada-002"


def test_openai_provider_uses_new_default_when_opted_in() -> None:
    provider = OpenAIEmbeddingProvider({"use_new_default_model": True})
    assert provider.model == "text-embedding-3-small"


def test_openai_provider_respects_explicit_model_setting() -> None:
    provider = OpenAIEmbeddingProvider({"model": "text-embedding-3-large"})
    assert provider.model == "text-embedding-3-large"
