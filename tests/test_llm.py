from app.graph import llm as llm_mod


def test_model_lists_are_comma_separated(monkeypatch):
    from app.config import Settings
    s = Settings(_env_file=None, google_api_key="g", groq_api_key="q",
                 gemini_model="gemini-a, gemini-b", groq_model="openai/gpt-oss-120b")
    monkeypatch.setattr(llm_mod, "get_settings", lambda: s)
    models = llm_mod.default_models()
    assert [getattr(m, "model", getattr(m, "model_name", None)) for m in models][-1] == "openai/gpt-oss-120b"
    assert len(models) == 3
    assert type(models[0]).__name__ == "ChatGoogleGenerativeAI"


def test_names_ignores_blanks():
    assert llm_mod._names(" a, ,b ,") == ["a", "b"]
