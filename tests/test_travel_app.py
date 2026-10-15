"""Streamlit smoke test for the standalone travel-planner form."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_travel_form_loads_without_api_credentials(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app_file = Path(__file__).resolve().parents[1] / "travel_app.py"

    app = AppTest.from_file(str(app_file), default_timeout=20).run()

    assert not app.exception
    assert any("Destination" in element.label for element in app.text_input)
    assert any("Total budget" in element.label for element in app.number_input)
    assert any("mock/unavailable" in element.value for element in app.info)
