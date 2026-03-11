from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.models.schemas import CandidateProfile, ReviewDecision
from app.utils.config import ROOT_DIR as APP_ROOT_DIR, settings

st.set_page_config(page_title="SignalDraft", layout="wide")

API_BASE_URL = settings.api_base_url.rstrip("/")
DATASET_PATH = APP_ROOT_DIR / "data" / "eval_dataset.json"


def load_demo_messages() -> list[dict[str, Any]]:
    items = json.loads(DATASET_PATH.read_text())
    return items[:3]


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def api_put(path: str, payload: dict[str, Any]) -> Any:
    response = requests.put(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def set_current_run(run: dict[str, Any]) -> None:
    st.session_state["current_run"] = run


def render_card(title: str, value: str, tone: str = "neutral") -> None:
    tone_class = f"sd-card {tone}"
    st.markdown(
        f"""
        <div class="{tone_class}">
            <div class="sd-card-label">{title}</div>
            <div class="sd-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_extracted_fields(run: dict[str, Any]) -> None:
    extracted = {key: value for key, value in run.get("extracted", {}).items() if value not in (None, "", [], False)}
    if not extracted:
        st.info("No structured fields were extracted from this message.")
        return
    html = "".join(
        f'<div class="sd-field-row"><span class="sd-field-key">{key.replace("_", " ").title()}</span>'
        f'<span class="sd-field-value">{", ".join(value) if isinstance(value, list) else value}</span></div>'
        for key, value in extracted.items()
    )
    st.markdown(f'<div class="sd-panel">{html}</div>', unsafe_allow_html=True)


def render_workflow_steps(run: dict[str, Any]) -> None:
    steps = run.get("workflow_steps", [])
    if not steps:
        st.caption("No workflow steps recorded yet.")
        return
    html = ""
    for step in steps:
        status = step.get("status", "completed")
        html += (
            f'<div class="sd-step {status}">'
            f'<div class="sd-step-name">{step.get("name", "")}</div>'
            f'<div class="sd-step-summary">{step.get("summary", "")}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="sd-steps">{html}</div>', unsafe_allow_html=True)


def sidebar_runs() -> None:
    with st.sidebar:
        st.markdown("## Past Runs")
        try:
            runs = api_get("/runs").get("items", [])
        except Exception as exc:
            st.error(f"Backend unavailable: {exc}")
            runs = []
        for item in runs:
            label = f"{item['message_type']} • {item['recommended_action']}"
            if st.button(label, key=f"run-{item['run_id']}", use_container_width=True):
                selected = api_get(f"/runs/{item['run_id']}")
                set_current_run(selected)


def profile_editor() -> None:
    st.markdown("### Candidate Profile")
    try:
        current_profile = CandidateProfile.model_validate(api_get("/profile"))
    except Exception as exc:
        st.error(f"Could not load profile: {exc}")
        return
    with st.form("profile_form"):
        full_name = st.text_input("Full name", current_profile.full_name)
        email = st.text_input("Email", current_profile.email)
        university = st.text_input("University", current_profile.university)
        graduation_date = st.text_input("Graduation date", current_profile.graduation_date)
        resume_summary = st.text_area("Resume summary", current_profile.resume_summary, height=120)
        preferred_tone = st.selectbox("Preferred tone", ["formal", "warm", "concise"], index=["formal", "warm", "concise"].index(current_profile.preferred_tone))
        target_roles = st.text_input("Target roles (comma-separated)", ", ".join(current_profile.target_roles))
        location = st.text_input("Location", current_profile.location)
        sponsorship_status = st.text_input("Sponsorship status", current_profile.sponsorship_status)
        portfolio_links = st.text_area("Portfolio links (one per line)", "\n".join(current_profile.portfolio_links), height=90)
        calendar_preferences = st.text_area("Calendar preferences", current_profile.calendar_preferences, height=90)
        default_signoff = st.text_input("Default signoff", current_profile.default_signoff)
        if st.form_submit_button("Save Profile", use_container_width=True):
            payload = {
                "full_name": full_name,
                "email": email,
                "university": university,
                "graduation_date": graduation_date,
                "resume_summary": resume_summary,
                "preferred_tone": preferred_tone,
                "target_roles": [item.strip() for item in target_roles.split(",") if item.strip()],
                "location": location,
                "sponsorship_status": sponsorship_status,
                "portfolio_links": [item.strip() for item in portfolio_links.splitlines() if item.strip()],
                "calendar_preferences": calendar_preferences,
                "default_signoff": default_signoff,
            }
            try:
                api_put("/profile", payload)
                st.success("Candidate profile updated.")
            except Exception as exc:
                st.error(f"Could not save profile: {exc}")


def analysis_workspace() -> None:
    demos = load_demo_messages()
    st.markdown(
        """
        <div class="sd-hero">
            <div class="sd-eyebrow">Local-first AI workflow</div>
            <h1>SignalDraft</h1>
            <p>Triage recruiter emails, interview updates, and networking replies into safe next actions and polished draft responses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("FastAPI + LangGraph + LangChain + LangSmith-ready tracing + SQLite + Streamlit.")

    demo_cols = st.columns(3)
    for idx, example in enumerate(demos):
        with demo_cols[idx]:
            if st.button(example["title"], use_container_width=True):
                st.session_state["message_input"] = example["message"]

    with st.form("analysis_form"):
        raw_message = st.text_area(
            "Paste inbound message",
            value=st.session_state.get("message_input", ""),
            height=260,
            placeholder="Paste a recruiter email, interview invite, scheduling request, or networking reply here.",
        )
        submitted = st.form_submit_button("Run Analysis", use_container_width=True)
        if submitted:
            try:
                run = api_post("/analyze", {"raw_message": raw_message})
                set_current_run(run)
                st.session_state["message_input"] = raw_message
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


def result_view() -> None:
    run = st.session_state.get("current_run")
    if not run:
        st.info("Run an analysis or select a past run to view results.")
        return

    current_run_id = run.get("run_id", "")
    if st.session_state.get("draft_editor_run_id") != current_run_id:
        st.session_state["draft_editor"] = run.get("draft_reply", "")
        st.session_state["draft_editor_run_id"] = current_run_id

    if run.get("needs_human_review"):
        st.warning(f"Needs human review: {run.get('review_reason') or 'Sensitive content detected.'}")

    metrics = st.columns(4)
    with metrics[0]:
        render_card("Category", run.get("message_type", "unknown"), "neutral")
    with metrics[1]:
        render_card("Urgency", run.get("urgency", "low"), "accent")
    with metrics[2]:
        render_card("Action", run.get("recommended_action", "draft_reply"), "success")
    with metrics[3]:
        render_card("Status", run.get("status", "analyzed"), "warning" if run.get("needs_human_review") else "neutral")

    st.markdown("### Why This Decision")
    st.markdown(f'<div class="sd-panel">{run.get("explanation", "No explanation generated.")}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### Extracted Fields")
        render_extracted_fields(run)
        st.markdown("### Workflow Steps")
        render_workflow_steps(run)
    with right:
        st.markdown("### Draft Reply")
        draft_value = st.text_area(
            "Editable draft",
            height=280,
            key="draft_editor",
        )
        action_cols = st.columns(3)
        notes = st.text_input("Review notes", value=run.get("review_notes", ""))
        with action_cols[0]:
            if st.button("Approve Draft", use_container_width=True):
                try:
                    updated = api_post(
                        f"/runs/{run['run_id']}/review",
                        {"decision": ReviewDecision.approved.value, "edited_draft": draft_value, "notes": notes},
                    )
                    set_current_run(updated)
                    st.success("Draft approved.")
                except Exception as exc:
                    st.error(f"Could not approve draft: {exc}")
        with action_cols[1]:
            if st.button("Reject Draft", use_container_width=True):
                try:
                    updated = api_post(
                        f"/runs/{run['run_id']}/review",
                        {"decision": ReviewDecision.rejected.value, "edited_draft": draft_value, "notes": notes},
                    )
                    set_current_run(updated)
                    st.warning("Draft rejected.")
                except Exception as exc:
                    st.error(f"Could not reject draft: {exc}")
        with action_cols[2]:
            if st.button("Mock Send", use_container_width=True):
                try:
                    updated = api_post(
                        f"/runs/{run['run_id']}/mock-send",
                        {"edited_draft": draft_value},
                    )
                    set_current_run(updated)
                    st.success("Mock send recorded.")
                except Exception as exc:
                    st.error(f"Could not record mock send: {exc}")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(27, 189, 160, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(244, 187, 68, 0.14), transparent 25%),
                linear-gradient(180deg, #f7faf9 0%, #f2f5f4 100%);
            color: #16302b;
        }
        .sd-hero {
            padding: 1.5rem 1.75rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #16302b 0%, #24453e 55%, #2f6f62 100%);
            color: #f7faf9;
            box-shadow: 0 18px 40px rgba(22, 48, 43, 0.15);
            margin-bottom: 0.8rem;
        }
        .sd-hero h1 {
            margin: 0;
            font-size: 2.4rem;
            letter-spacing: -0.04em;
        }
        .sd-hero p {
            margin: 0.6rem 0 0;
            max-width: 680px;
            font-size: 1rem;
            color: rgba(247, 250, 249, 0.92);
        }
        .sd-eyebrow {
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            opacity: 0.78;
            margin-bottom: 0.6rem;
        }
        .sd-card {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: white;
            border: 1px solid rgba(22, 48, 43, 0.08);
            min-height: 110px;
            box-shadow: 0 10px 22px rgba(16, 39, 35, 0.08);
        }
        .sd-card.accent {
            background: linear-gradient(180deg, #e6f5f1 0%, white 100%);
        }
        .sd-card.success {
            background: linear-gradient(180deg, #eefaf5 0%, white 100%);
        }
        .sd-card.warning {
            background: linear-gradient(180deg, #fff6e8 0%, white 100%);
        }
        .sd-card-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6a7d78;
            margin-bottom: 0.5rem;
        }
        .sd-card-value {
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.3;
            color: #16302b;
        }
        .sd-panel {
            background: white;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(22, 48, 43, 0.08);
            box-shadow: 0 8px 18px rgba(16, 39, 35, 0.06);
        }
        .sd-field-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.6rem 0;
            border-bottom: 1px solid rgba(22, 48, 43, 0.08);
        }
        .sd-field-row:last-child {
            border-bottom: 0;
        }
        .sd-field-key {
            color: #5e716c;
            font-size: 0.9rem;
        }
        .sd-field-value {
            color: #16302b;
            text-align: right;
            font-weight: 600;
        }
        .sd-steps {
            display: grid;
            gap: 0.7rem;
        }
        .sd-step {
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: white;
            border: 1px solid rgba(22, 48, 43, 0.08);
            box-shadow: 0 8px 18px rgba(16, 39, 35, 0.06);
        }
        .sd-step.warning {
            border-color: rgba(229, 154, 51, 0.35);
            background: #fff8ed;
        }
        .sd-step-name {
            font-weight: 700;
            color: #16302b;
            margin-bottom: 0.3rem;
        }
        .sd-step-summary {
            color: #5e716c;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    sidebar_runs()
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        analysis_workspace()
        st.divider()
        profile_editor()
    with right:
        result_view()


if __name__ == "__main__":
    main()
