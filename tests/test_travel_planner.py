"""Focused tests for travel request validation and deterministic planning utilities."""

from datetime import date

import pytest

from travel_planner import (
    TravelPlanningError,
    TripRequest,
    _get_weather,
    _tavily_search,
    calculate_budget,
    create_day_schedule,
)


def make_trip(**overrides):
    values = {
        "destination": "Kyoto, Japan",
        "starting_location": "Seattle, USA",
        "start_date": date(2027, 4, 10),
        "end_date": date(2027, 4, 12),
        "budget": 1000,
        "interests": ("Culture", "Food"),
    }
    values.update(overrides)
    return TripRequest(**values)


def test_trip_duration_counts_calendar_days_and_nights():
    trip = make_trip()

    assert trip.days == 3
    assert trip.nights == 2
    assert trip.as_prompt_data()["interests"] == ["Culture", "Food"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"destination": " "}, "destination"),
        ({"starting_location": ""}, "starting location"),
        ({"end_date": date(2027, 4, 9)}, "return date"),
        ({"budget": 0}, "budget"),
        ({"budget": float("nan")}, "budget"),
        ({"budget": float("inf")}, "budget"),
        ({"interests": ()}, "activity"),
        ({"end_date": date(2027, 6, 20)}, "61 calendar days"),
    ],
)
def test_trip_request_rejects_invalid_requirements(overrides, message):
    with pytest.raises(TravelPlanningError, match=message):
        make_trip(**overrides)


def test_budget_is_itemized_and_compares_exactly_to_budget():
    trip = make_trip(budget=660)

    estimate = calculate_budget(trip)

    assert estimate.line_items == {
        "Accommodation": 240,
        "Meals": 135,
        "Local transport": 60,
        "Activities": 105,
        "Long-distance transport": 250,
    }
    assert estimate.total == 790
    assert estimate.remaining == -130
    assert not estimate.within_budget


def test_budget_can_be_recalculated_with_lower_cost_assumptions():
    estimate = calculate_budget(
        make_trip(budget=500),
        hotel_per_night=80,
        meals_per_day=35,
        local_transport_per_day=15,
        activities_per_day=15,
        long_distance_transport=200,
    )

    assert estimate.total == 555
    assert not estimate.within_budget


def test_schedule_contains_a_slot_for_each_dated_trip_day():
    schedule = create_day_schedule(date(2027, 4, 10), 2, ("Nature",))

    assert len(schedule) == 2
    assert "2027-04-10" in schedule[0]
    assert "2027-04-11" in schedule[1]
    assert "Nature" in schedule[0]


def test_search_without_credentials_is_explicitly_marked_as_mock(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = _tavily_search("attractions in Kyoto")

    assert result["status"] == "mock"
    assert result["results"] == []
    assert "not configured" in result["message"]


def test_weather_outside_forecast_window_does_not_claim_live_data():
    result = _get_weather("Kyoto, Japan", date(2027, 4, 10), date(2027, 4, 12))

    assert result["status"] == "unavailable"
    assert result["sources"]
    assert "forecast window" in result["message"]
