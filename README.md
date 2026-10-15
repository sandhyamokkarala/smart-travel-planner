# Smart Travel Planner

A standalone multi-agent travel planner built with Python, CrewAI, Streamlit, and Ollama or
OpenRouter. Enter a destination, starting location, dates, total budget, and interests to generate
a researched day-by-day itinerary. CrewAI agents coordinate destination research, accommodation
and transport, expense estimates, itinerary planning, and a final budget review. When the initial
estimate exceeds the budget, the workflow performs a lower-cost revision.

## Run on Windows

Use a dedicated environment for this project:

```powershell
cd $HOME\Downloads\smart-travel-planner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a local Ollama model, start the Ollama service and pull a model, then configure and launch:

```powershell
ollama pull llama3.1
$env:TRAVEL_LLM_PROVIDER = "ollama"
$env:TRAVEL_LLM_MODEL = "llama3.1"
streamlit run travel_app.py
```

For OpenRouter instead, set `TRAVEL_LLM_PROVIDER` to `openrouter`, provide `OPENROUTER_API_KEY`,
and set `TRAVEL_LLM_MODEL` to an available model identifier, such as
`deepseek/deepseek-chat-v3-0324`.

## Research providers and estimates

Set `TAVILY_API_KEY` to enable live web searches for attractions, accommodation, and transport.
Without it, these searches are explicitly marked as mock/unavailable; the planner does not invent
search results. Weather uses the public Open-Meteo forecast/geocoding API when the whole trip is
within the forecast window. Road distances use the public OSRM demo routing service. These public
services may be unavailable or rate-limited.

Budget results are illustrative USD allowances for one traveller, not live booking quotes or a
guarantee of availability. Confirm prices and travel arrangements with providers before booking.

## Tests

```powershell
python -m pytest tests -q
```

The CrewAI workflow construction test is skipped when CrewAI is not installed. The tests do not
require live API credentials or an LLM service.
