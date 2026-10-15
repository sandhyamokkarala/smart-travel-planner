"""CrewAI travel planning workflow and its external research tools."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Callable

import requests

REQUEST_TIMEOUT = 15
OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_SOURCE = "https://open-meteo.com/en/docs"
OSRM_SOURCE = "https://project-osrm.org/"
TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TravelPlanningError(ValueError):
    """An actionable validation or planning configuration error."""


class ToolIntegrationError(RuntimeError):
    """An external travel-data provider failed to return usable data."""


@dataclass(frozen=True)
class TripRequest:
    destination: str
    starting_location: str
    start_date: date
    end_date: date
    budget: float
    interests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.destination.strip():
            raise TravelPlanningError("Enter a travel destination.")
        if not self.starting_location.strip():
            raise TravelPlanningError("Enter a starting location.")
        if self.end_date < self.start_date:
            raise TravelPlanningError("The return date must be on or after the start date.")
        if (self.end_date - self.start_date).days > 60:
            raise TravelPlanningError("Trips can be at most 61 calendar days long.")
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise TravelPlanningError("The total budget must be greater than zero.")
        if not self.interests:
            raise TravelPlanningError("Select at least one preferred activity.")
        if any(not interest.strip() for interest in self.interests):
            raise TravelPlanningError("Preferred activities cannot be blank.")

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days

    def as_prompt_data(self) -> dict[str, Any]:
        return {
            "destination": self.destination.strip(),
            "starting_location": self.starting_location.strip(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "days": self.days,
            "nights": self.nights,
            "budget": self.budget,
            "interests": list(self.interests),
        }


@dataclass(frozen=True)
class BudgetEstimate:
    currency: str
    line_items: dict[str, float]
    total: float
    budget: float
    remaining: float
    within_budget: bool
    assumptions: str


@dataclass(frozen=True)
class TravelPlanResult:
    itinerary: str
    budget: BudgetEstimate
    lower_cost_alternative: BudgetEstimate | None
    research: list[dict[str, Any]]
    sources: list[str]


def calculate_budget(
    trip: TripRequest,
    *,
    hotel_per_night: float = 120,
    meals_per_day: float = 45,
    local_transport_per_day: float = 20,
    activities_per_day: float = 35,
    long_distance_transport: float = 250,
) -> BudgetEstimate:
    """Calculate a transparent single-traveller planning allowance in USD."""
    rates = {
        "Accommodation": (hotel_per_night, trip.nights),
        "Meals": (meals_per_day, trip.days),
        "Local transport": (local_transport_per_day, trip.days),
        "Activities": (activities_per_day, trip.days),
        "Long-distance transport": (long_distance_transport, 1),
    }
    if any(not math.isfinite(rate) or rate < 0 for rate, _ in rates.values()):
        raise TravelPlanningError("Cost assumptions must be finite and cannot be negative.")
    line_items = {name: round(rate * count, 2) for name, (rate, count) in rates.items()}
    total = round(sum(line_items.values()), 2)
    return BudgetEstimate(
        currency="USD",
        line_items=line_items,
        total=total,
        budget=round(trip.budget, 2),
        remaining=round(trip.budget - total, 2),
        within_budget=total <= trip.budget,
        assumptions=(
            "Illustrative per-person planning allowances, not live booking quotes: "
            f"${hotel_per_night:.0f}/night lodging, ${meals_per_day:.0f}/day meals, "
            f"${local_transport_per_day:.0f}/day local transport, "
            f"${activities_per_day:.0f}/day activities, and "
            f"${long_distance_transport:.0f} return long-distance transport. "
            "Confirm current prices before booking."
        ),
    )


def create_day_schedule(start_date: date, days: int, interests: tuple[str, ...]) -> list[str]:
    """Return dated itinerary slots for the itinerary agent to fill."""
    if days < 1 or days > 61:
        raise TravelPlanningError("The itinerary must cover between 1 and 61 days.")
    preferences = ", ".join(interests)
    return [
        f"Day {index + 1} — {(start_date + timedelta(days=index)).isoformat()}: "
        f"plan a morning, afternoon, and evening around {preferences}; "
        "include meal breaks and realistic travel time."
        for index in range(days)
    ]


def _request_json(
    method: str, url: str, *, params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ToolIntegrationError(f"Request to {url} failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ToolIntegrationError(f"Request to {url} returned an invalid response.")
    return result


def _tavily_search(query: str) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "status": "mock",
            "query": query,
            "message": "TAVILY_API_KEY is not configured; no live search results were retrieved.",
            "results": [],
            "sources": [],
        }
    response = _request_json(
        "POST",
        TAVILY_ENDPOINT,
        payload={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5},
    )
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in response.get("results", [])
    ]
    return {
        "status": "live",
        "query": query,
        "results": results,
        "sources": [item["url"] for item in results if item["url"]],
    }


def _geocode(place: str) -> dict[str, Any]:
    response = _request_json(
        "GET",
        OPEN_METEO_GEOCODING,
        params={"name": place, "count": 1, "language": "en", "format": "json"},
    )
    results = response.get("results", [])
    if not results:
        raise ToolIntegrationError(f"No location match found for {place!r}.")
    return results[0]


def _search_destination(destination: str, interests: tuple[str, ...]) -> dict[str, Any]:
    query = f"Top attractions, local experiences, and visitor information in {destination}; "
    query += "interests: " + ", ".join(interests)
    return _tavily_search(query)


def _search_stays(destination: str, nights: int) -> dict[str, Any]:
    return _tavily_search(
        f"Hotel and accommodation options in {destination} for {nights} nights; "
        "include current indicative prices and official booking links"
    )


def _search_transport(origin: str, destination: str, start: date, end: date) -> dict[str, Any]:
    return _tavily_search(
        f"Transport options from {origin} to {destination}, departing {start.isoformat()} "
        f"and returning {end.isoformat()}; include current fares and operator links"
    )


def _get_weather(destination: str, start: date, end: date) -> dict[str, Any]:
    today = date.today()
    last_forecast_day = today + timedelta(days=15)
    if start < today or end > last_forecast_day:
        return {
            "status": "unavailable",
            "message": (
                f"Open-Meteo forecasts are requested only when the complete trip is within "
                f"the forecast window ({today.isoformat()} through {last_forecast_day.isoformat()}); "
                "check a forecast closer to departure."
            ),
            "sources": [OPEN_METEO_SOURCE],
        }
    location = _geocode(destination)
    response = _request_json(
        "GET",
        OPEN_METEO_FORECAST,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "auto",
        },
    )
    return {
        "status": "live",
        "location": location.get("name", destination),
        "daily": response.get("daily", {}),
        "sources": [OPEN_METEO_SOURCE],
    }


def _estimate_route(origin: str, destination: str) -> dict[str, Any]:
    start = _geocode(origin)
    end = _geocode(destination)
    response = _request_json(
        "GET",
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start['longitude']},{start['latitude']};{end['longitude']},{end['latitude']}",
        params={"overview": "false"},
    )
    routes = response.get("routes", [])
    if response.get("code") != "Ok" or not routes:
        raise ToolIntegrationError(f"No road route found from {origin!r} to {destination!r}.")
    route = routes[0]
    return {
        "status": "live",
        "distance_km": round(route["distance"] / 1000, 1),
        "driving_hours": round(route["duration"] / 3600, 1),
        "message": "Road-route estimate; it does not account for all border, ferry, or flight options.",
        "sources": [OSRM_SOURCE],
    }


def _collect_research(trip: TripRequest) -> list[dict[str, Any]]:
    lookups: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("Destination search (Tavily)", lambda: _search_destination(trip.destination, trip.interests)),
        ("Accommodation search (Tavily)", lambda: _search_stays(trip.destination, trip.nights)),
        (
            "Transport search (Tavily)",
            lambda: _search_transport(
                trip.starting_location, trip.destination, trip.start_date, trip.end_date
            ),
        ),
        ("Weather forecast (Open-Meteo)", lambda: _get_weather(trip.destination, trip.start_date, trip.end_date)),
        ("Road routing (OSRM)", lambda: _estimate_route(trip.starting_location, trip.destination)),
    )
    findings: list[dict[str, Any]] = []
    for name, lookup in lookups:
        try:
            finding = lookup()
        except ToolIntegrationError as exc:
            finding = {"status": "unavailable", "message": str(exc), "sources": []}
        findings.append({"tool": name, **finding})
    return findings


def _sources_from(research: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            source
            for finding in research
            for source in finding.get("sources", [])
            if isinstance(source, str) and source.startswith(("https://", "http://"))
        )
    )


def _build_crew(trip: TripRequest, research: list[dict[str, Any]], estimate: BudgetEstimate):
    try:
        from crewai import Agent, Crew, LLM, Process, Task
        from crewai.tools import tool
    except ImportError as exc:
        raise TravelPlanningError(
            "CrewAI is not installed. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    evidence_by_tool = {finding["tool"]: finding for finding in research}

    def evidence_json(tool_name: str) -> str:
        return json.dumps(evidence_by_tool[tool_name], ensure_ascii=False)

    provider = os.getenv("TRAVEL_LLM_PROVIDER", "ollama").strip().lower()
    requested_model = os.getenv("TRAVEL_LLM_MODEL", "llama3.1").strip()
    if not requested_model:
        raise TravelPlanningError("TRAVEL_LLM_MODEL cannot be blank.")
    if provider == "ollama":
        model = requested_model if requested_model.startswith("ollama/") else f"ollama/{requested_model}"
        llm = LLM(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise TravelPlanningError(
                "Set OPENROUTER_API_KEY before selecting the OpenRouter provider."
            )
        model = requested_model.removeprefix("openrouter/")
        llm = LLM(model=f"openrouter/{model}", api_key=api_key)
    else:
        raise TravelPlanningError("TRAVEL_LLM_PROVIDER must be either 'ollama' or 'openrouter'.")

    @tool("Search destination information")
    def destination_search() -> str:
        """Search for current attractions and local experiences for the requested trip."""
        return evidence_json("Destination search (Tavily)")

    @tool("Search accommodation options")
    def accommodation_search() -> str:
        """Search for hotels and accommodation in the destination."""
        return evidence_json("Accommodation search (Tavily)")

    @tool("Search transport options")
    def transport_search() -> str:
        """Search transport options between the starting location and destination."""
        return evidence_json("Transport search (Tavily)")

    @tool("Get destination weather forecast")
    def weather_forecast() -> str:
        """Retrieve a date-specific forecast if the complete trip is inside the forecast window."""
        return evidence_json("Weather forecast (Open-Meteo)")

    @tool("Estimate road route")
    def road_route() -> str:
        """Estimate road distance and driving time between the two locations."""
        return evidence_json("Road routing (OSRM)")

    @tool("Calculate trip expenses")
    def expense_calculator(
        hotel_per_night: float = 120,
        meals_per_day: float = 45,
        local_transport_per_day: float = 20,
        activities_per_day: float = 35,
        long_distance_transport: float = 250,
    ) -> str:
        """Calculate itemized USD allowances for this trip using the stated rate assumptions."""
        result = calculate_budget(
            trip,
            hotel_per_night=hotel_per_night,
            meals_per_day=meals_per_day,
            local_transport_per_day=local_transport_per_day,
            activities_per_day=activities_per_day,
            long_distance_transport=long_distance_transport,
        )
        return json.dumps(asdict(result))

    @tool("Build dated daily schedule")
    def schedule_builder() -> str:
        """Create one dated itinerary slot for each travel day."""
        return json.dumps(
            create_day_schedule(trip.start_date, trip.days, trip.interests),
            ensure_ascii=False,
        )

    coordinator = Agent(
        role="Travel Coordinator",
        goal="Coordinate specialist research, enforce the customer's constraints, and present a sourced plan.",
        backstory="You are a careful trip coordinator who distinguishes verified live information from estimates.",
        llm=llm,
        verbose=False,
        allow_delegation=True,
    )
    researcher = Agent(
        role="Destination Research Agent",
        goal="Find interest-matched attractions, local experiences, and relevant weather information.",
        backstory="You research destinations and cite current results rather than inventing facts.",
        tools=[destination_search, weather_forecast],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    hotel_transport = Agent(
        role="Hotel and Transport Agent",
        goal="Find accommodation and transport options and assess route feasibility.",
        backstory="You compare travel logistics and clearly label live search results versus planning estimates.",
        tools=[accommodation_search, transport_search, road_route],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    analyst = Agent(
        role="Budget Analyst",
        goal="Calculate trip costs transparently and check each plan against the customer's total budget.",
        backstory="You are a numerical analyst who uses the calculator and never presents assumptions as quotes.",
        tools=[expense_calculator],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    planner = Agent(
        role="Itinerary Planner",
        goal="Create a practical date-specific day-by-day itinerary from approved research and budget findings.",
        backstory="You build realistic schedules with meal breaks, travel time, and interest-matched alternatives.",
        tools=[schedule_builder],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    trip_data = json.dumps(trip.as_prompt_data(), ensure_ascii=False)
    research_data = json.dumps(research, ensure_ascii=False)
    budget_data = json.dumps(asdict(estimate), ensure_ascii=False)
    intake_task = Task(
        description=(
            "Review the customer's travel requirements {trip_request}. Summarize destination, "
            "origin, dates, trip length, budget ceiling, and interests as constraints for the "
            "specialists. Do not invent missing preferences or costs."
        ),
        expected_output="A concise, constraint-focused brief for the destination, logistics, budget, and itinerary agents.",
        agent=coordinator,
    )
    research_task = Task(
        description=(
            "Research this trip: {trip_request}. Use destination_search and weather_forecast. "
            "Use only supported facts from tool results, cite URLs, and explicitly label missing "
            "or out-of-window forecasts. Research evidence collected by the application: {evidence}."
        ),
        expected_output="A concise list of sourced attractions, local experiences, and date-specific weather status.",
        agent=researcher,
        context=[intake_task],
    )
    logistics_task = Task(
        description=(
            "Find accommodation and transport for {trip_request}. Use accommodation_search, "
            "transport_search, and road_route. Clearly identify missing search credentials, "
            "unavailable data, and any road-routing limitations. Evidence: {evidence}."
        ),
        expected_output="Sourced accommodation and transport options with availability/price caveats.",
        agent=hotel_transport,
        context=[intake_task],
    )
    budget_task = Task(
        description=(
            "Review the research and logistics outputs. Call expense_calculator using its stated "
            "default per-person allowances, then compare with the exact deterministic estimate "
            "{budget}. Do not call allowances booking quotes. Recommend cost changes if over budget."
        ),
        expected_output="An itemized USD budget calculation and an explicit within-budget decision.",
        agent=analyst,
        context=[intake_task, research_task, logistics_task],
    )
    itinerary_task = Task(
        description=(
            "Build the itinerary for {trip_request} using the research, logistics, budget, and "
            "dated schedule tool. Respect the budget calculation {budget}; include day-by-day "
            "morning/afternoon/evening suggestions, practical transit, and sources."
        ),
        expected_output="A practical dated day-by-day itinerary with hotels, transport, estimated costs, and source URLs.",
        agent=planner,
        context=[intake_task, research_task, logistics_task, budget_task],
    )
    coordination_task = Task(
        description=(
            "Present the complete personalized travel plan for {trip_request}. Reconcile all "
            "specialist outputs and use the exact budget figures {budget}. Distinguish live "
            "searches from illustrative estimates, state whether the budget is met, and list "
            "source URLs. Do not invent availability, prices, or citations."
        ),
        expected_output="A complete customer-ready travel plan and explicit budget decision.",
        agent=coordinator,
        context=[intake_task, research_task, logistics_task, budget_task, itinerary_task],
    )
    crew = Crew(
        agents=[coordinator, researcher, hotel_transport, analyst, planner],
        tasks=[
            intake_task,
            research_task,
            logistics_task,
            budget_task,
            itinerary_task,
            coordination_task,
        ],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff(
        inputs={"trip_request": trip_data, "evidence": research_data, "budget": budget_data}
    )
    return result, (hotel_transport, analyst, planner, coordinator)


def _run_budget_revision(
    trip: TripRequest,
    cheaper_estimate: BudgetEstimate,
    research: list[dict[str, Any]],
    crew_parts: tuple[Any, Any, Any, Any],
    first_result: Any,
) -> str:
    from crewai import Crew, Process, Task

    hotel_transport, analyst, planner, coordinator = crew_parts
    revised_budget = json.dumps(asdict(cheaper_estimate), ensure_ascii=False)
    revise_logistics = Task(
        description=(
            "The original trip exceeded the customer's budget. Find lower-cost accommodation "
            "and transport alternatives using your search tools and preserve the trip dates and "
            "interests. The revised calculator result is {revised_budget}. Explicitly distinguish "
            "live options from fallback allowances. The trip requirements are {trip_request}. "
            "Original coordination result: {prior_plan}. Additional research evidence: {evidence}."
        ),
        expected_output="A revised low-cost accommodation and transport plan with sourced alternatives.",
        agent=hotel_transport,
    )
    revise_budget = Task(
        description=(
            "Call expense_calculator using these lower planning allowances, verify the revised "
            "rates (hotel_per_night=80, meals_per_day=35, local_transport_per_day=15, "
            "activities_per_day=15, long_distance_transport=200), compare with the exact "
            "revised total {revised_budget}, and state whether it is within the trip budget. "
            "Keep all figures labelled as estimates."
        ),
        expected_output="An itemized revised USD budget calculation and explicit budget comparison.",
        agent=analyst,
        context=[revise_logistics],
    )
    revise_itinerary = Task(
        description=(
            "Revise the dated itinerary to fit the lower-cost alternatives and budget {revised_budget}. "
            "Preserve the required destination, dates, and interests; use schedule_builder."
        ),
        expected_output="A revised daily itinerary with budget-friendly choices and transport details.",
        agent=planner,
        context=[revise_logistics, revise_budget],
    )
    final_task = Task(
        description=(
            "Present the revised travel plan for {trip_request}. Use the lower-cost logistics and "
            "the exact deterministic budget {revised_budget}; explicitly state if even this "
            "alternative exceeds the budget and identify what the customer should change. "
            "Cite sources and distinguish estimates from live prices."
        ),
        expected_output="A complete revised travel plan with a truthful budget comparison and sources.",
        agent=coordinator,
        context=[revise_logistics, revise_budget, revise_itinerary],
    )
    result = Crew(
        agents=[hotel_transport, analyst, planner, coordinator],
        tasks=[revise_logistics, revise_budget, revise_itinerary, final_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(
        inputs={
            "trip_request": json.dumps(trip.as_prompt_data(), ensure_ascii=False),
            "evidence": json.dumps(research, ensure_ascii=False),
            "prior_plan": str(first_result),
            "revised_budget": revised_budget,
        }
    )
    return str(result)


def create_travel_plan(trip: TripRequest) -> TravelPlanResult:
    """Collect provider evidence, run the CrewAI workflow, and revise over-budget plans."""
    research = _collect_research(trip)
    estimate = calculate_budget(trip)
    first_result, crew_parts = _build_crew(trip, research, estimate)
    itinerary = str(first_result)
    lower_cost = None
    if not estimate.within_budget:
        lower_cost = calculate_budget(
            trip,
            hotel_per_night=80,
            meals_per_day=35,
            local_transport_per_day=15,
            activities_per_day=15,
            long_distance_transport=200,
        )
        itinerary = _run_budget_revision(
            trip, lower_cost, research, crew_parts, first_result
        )
    return TravelPlanResult(
        itinerary=itinerary,
        budget=estimate,
        lower_cost_alternative=lower_cost,
        research=research,
        sources=_sources_from(research),
    )
