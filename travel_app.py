"""Standalone Streamlit interface for the CrewAI travel planner."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import streamlit as st

from travel_planner import (
    BudgetEstimate,
    TravelPlanResult,
    TripRequest,
    TravelPlanningError,
    create_travel_plan,
)

logger = logging.getLogger(__name__)


def _show_budget(label: str, estimate: BudgetEstimate) -> None:
    st.subheader(label)
    first, second, third = st.columns(3)
    first.metric("Estimated total", f"${estimate.total:,.2f}")
    second.metric("Your budget", f"${estimate.budget:,.2f}")
    third.metric("Budget remaining", f"${estimate.remaining:,.2f}")
    if estimate.within_budget:
        st.success("This illustrative estimate is within budget.")
    else:
        st.error("This illustrative estimate exceeds budget.")
    st.dataframe(
        [{"Expense": name, "Estimated cost (USD)": cost} for name, cost in estimate.line_items.items()],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(estimate.assumptions)


def _show_result(result: TravelPlanResult) -> None:
    st.header("Your travel plan")
    st.markdown(result.itinerary)
    _show_budget("Budget analysis", result.budget)
    if result.lower_cost_alternative is not None:
        _show_budget("Revised lower-cost alternative", result.lower_cost_alternative)
        if not result.lower_cost_alternative.within_budget:
            st.warning(
                "The lower-cost estimate still exceeds the budget. Adjust the budget, dates, "
                "or destination before booking."
            )
    st.subheader("Research tool results")
    for finding in result.research:
        with st.expander(f"{finding['tool']} — {finding['status']}"):
            if finding.get("message"):
                st.write(finding["message"])
            if finding.get("error"):
                st.error(finding["error"])
            if finding.get("results"):
                for item in finding["results"]:
                    title = item.get("title") or "Search result"
                    url = item.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url}) — {item.get('content', '')}")
                    else:
                        st.write(f"- {title}: {item.get('content', '')}")
            if finding.get("daily"):
                st.json(finding["daily"])
            if finding.get("distance_km") is not None:
                st.write(
                    f"Road distance: {finding['distance_km']} km; "
                    f"estimated driving time: {finding['driving_hours']} hours."
                )
    if result.sources:
        st.subheader("Sources")
        for source in result.sources:
            st.markdown(f"- [{source}]({source})")
    else:
        st.info("No live web-search sources were returned. Review the tool statuses above.")


def main() -> None:
    st.set_page_config(page_title="CrewAI Travel Planner", page_icon="✈️", layout="wide")
    st.title("✈️ Smart Travel Planning")
    st.caption(
        "A multi-agent travel planner that researches options, checks the budget, "
        "and revises over-budget plans."
    )
    provider = os.getenv("TRAVEL_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "openrouter":
        st.caption("LLM provider: OpenRouter")
    else:
        st.caption(
            f"LLM provider: Ollama ({os.getenv('TRAVEL_LLM_MODEL', 'llama3.1')})"
        )
    if not os.getenv("TAVILY_API_KEY"):
        st.info(
            "TAVILY_API_KEY is not configured: destination, hotel, and transport web searches "
            "will be clearly marked as mock/unavailable. Weather and road routing use public APIs."
        )

    today = date.today()
    with st.form("travel_request"):
        destination_column, origin_column = st.columns(2)
        destination = destination_column.text_input("Destination", placeholder="Kyoto, Japan")
        starting_location = origin_column.text_input(
            "Starting location", placeholder="Seattle, USA"
        )
        date_column, budget_column = st.columns(2)
        start_date = date_column.date_input("Departure date", value=today + timedelta(days=30))
        end_date = date_column.date_input("Return date", value=today + timedelta(days=34))
        budget = budget_column.number_input(
            "Total budget (USD)", min_value=1.0, value=2000.0, step=100.0
        )
        interests = st.multiselect(
            "Preferred activities",
            ["Sightseeing", "Shopping", "Adventure", "Relaxation", "Food", "Culture", "Nature"],
            default=["Sightseeing", "Culture"],
        )
        submitted = st.form_submit_button("Plan my trip", type="primary")

    if submitted:
        try:
            trip = TripRequest(
                destination=destination,
                starting_location=starting_location,
                start_date=start_date,
                end_date=end_date,
                budget=budget,
                interests=tuple(interests),
            )
            with st.spinner("Researching options and coordinating the travel agents..."):
                st.session_state["travel_plan_result"] = create_travel_plan(trip)
        except TravelPlanningError as exc:
            st.error(str(exc))
        except Exception as exc:
            logger.exception("Travel planning failed")
            st.error(f"Travel planning failed: {exc}")

    result = st.session_state.get("travel_plan_result")
    if isinstance(result, TravelPlanResult):
        _show_result(result)


if __name__ == "__main__":
    main()
