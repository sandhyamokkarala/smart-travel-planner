"""CrewAI wiring checks that do not require an LLM or external API credentials."""

from datetime import date

import pytest

crewai = pytest.importorskip("crewai")

from travel_planner import (  # noqa: E402
    TripRequest,
    _build_crew,
    _run_budget_revision,
    calculate_budget,
    create_travel_plan,
)


def test_crew_tasks_and_budget_revision_are_constructed_without_llm_calls(monkeypatch):
    monkeypatch.setenv("TRAVEL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TRAVEL_LLM_MODEL", "llama3.1")
    runs = []

    def record_crew(self, inputs):
        runs.append(
            {
                "roles": [agent.role for agent in self.agents],
                "tasks": list(self.tasks),
            }
        )
        return "mock initial plan"

    monkeypatch.setattr(crewai.Crew, "kickoff", record_crew)
    trip = TripRequest(
        destination="Kyoto, Japan",
        starting_location="Seattle, USA",
        start_date=date(2027, 4, 10),
        end_date=date(2027, 4, 12),
        budget=300,
        interests=("Culture",),
    )
    evidence = [
        {"tool": name, "status": "mock", "results": [], "sources": []}
        for name in (
            "Destination search (Tavily)",
            "Accommodation search (Tavily)",
            "Transport search (Tavily)",
            "Weather forecast (Open-Meteo)",
            "Road routing (OSRM)",
        )
    ]

    first_result, agents = _build_crew(trip, evidence, calculate_budget(trip))
    revised = calculate_budget(
        trip,
        hotel_per_night=80,
        meals_per_day=35,
        local_transport_per_day=15,
        activities_per_day=15,
        long_distance_transport=200,
    )
    revised_result = _run_budget_revision(trip, revised, evidence, agents, first_result)

    assert first_result == "mock initial plan"
    assert revised_result == "mock initial plan"
    assert revised.total < calculate_budget(trip).total
    assert runs[0]["roles"] == [
        "Travel Coordinator",
        "Destination Research Agent",
        "Hotel and Transport Agent",
        "Budget Analyst",
        "Itinerary Planner",
    ]
    assert len(runs[0]["tasks"]) == 6
    assert all(task.agent is not None for task in runs[0]["tasks"])
    assert all(task.context for task in runs[0]["tasks"][1:])
    assert len(runs[1]["tasks"]) == 4
    assert all(task.context for task in runs[1]["tasks"][1:])


@pytest.mark.parametrize(("budget", "expect_revision"), [(2000, False), (300, True)])
def test_plan_orchestrator_revises_only_when_budget_is_exceeded(
    monkeypatch, budget, expect_revision
):
    trip = TripRequest(
        destination="Kyoto, Japan",
        starting_location="Seattle, USA",
        start_date=date(2027, 4, 10),
        end_date=date(2027, 4, 12),
        budget=budget,
        interests=("Culture",),
    )
    research = [{"tool": "test", "status": "mock", "sources": []}]
    crew_result = "initial plan"
    revision_calls = []
    monkeypatch.setattr("travel_planner._collect_research", lambda request: research)
    monkeypatch.setattr(
        "travel_planner._build_crew",
        lambda request, findings, estimate: (crew_result, ("hotel", "budget", "planner", "coordinator")),
    )
    monkeypatch.setattr(
        "travel_planner._run_budget_revision",
        lambda *args: revision_calls.append(args) or "revised plan",
    )

    result = create_travel_plan(trip)

    assert result.budget.within_budget is (not expect_revision)
    assert result.lower_cost_alternative is not None if expect_revision else result.lower_cost_alternative is None
    assert bool(revision_calls) is expect_revision
    assert result.itinerary == ("revised plan" if expect_revision else crew_result)
