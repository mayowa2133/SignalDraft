from __future__ import annotations

import hmac
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

API_BASE_URL = settings.resolved_api_base_url
DATASET_PATH = APP_ROOT_DIR / "data" / "eval_dataset.json"


def load_demo_messages() -> list[dict[str, Any]]:
    items = json.loads(DATASET_PATH.read_text())
    return items[:3]


def get_api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.api_token:
        headers["Authorization"] = f"Bearer {settings.api_token}"
    return headers


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Request failed with status {response.status_code}."

    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("code") or response.reason)
    if isinstance(detail, str):
        return detail
    return f"Request failed with status {response.status_code}."


def api_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.request(
        method=method,
        url=f"{API_BASE_URL}{path}",
        json=payload,
        headers=get_api_headers(),
        timeout=60,
    )
    if response.ok:
        return response.json()
    raise RuntimeError(extract_error_message(response))


def api_get(path: str) -> Any:
    return api_request("GET", path)


def api_post(path: str, payload: dict[str, Any]) -> Any:
    return api_request("POST", path, payload)


def api_put(path: str, payload: dict[str, Any]) -> Any:
    return api_request("PUT", path, payload)


def set_current_run(run: dict[str, Any]) -> None:
    st.session_state["current_run"] = run


def render_metric_card(title: str, value: str) -> None:
    with st.container(border=True):
        st.caption(title)
        st.write(f"**{value}**")


def render_extracted_fields(run: dict[str, Any]) -> None:
    extracted = {
        key.replace("_", " ").title(): ", ".join(value) if isinstance(value, list) else str(value)
        for key, value in run.get("extracted", {}).items()
        if value not in (None, "", [], False)
    }
    if not extracted:
        st.info("No structured fields were extracted from this message.")
        return

    rows = [{"Field": key, "Value": value} for key, value in extracted.items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_workflow_steps(run: dict[str, Any]) -> None:
    steps = run.get("workflow_steps", [])
    if not steps:
        st.caption("No workflow steps recorded yet.")
        return

    status_copy = {
        "completed": "Completed",
        "warning": "Needs review",
        "skipped": "Skipped",
    }
    for step in steps:
        with st.container(border=True):
            st.caption(status_copy.get(step.get("status", "completed"), "Completed"))
            st.write(f"**{step.get('name', '').replace('_', ' ').title()}**")
            st.write(step.get("summary", ""))


def load_readiness() -> dict[str, Any] | None:
    try:
        return api_get("/readiness")
    except Exception as exc:
        st.error(f"Could not load backend readiness: {exc}")
        return None


def render_readiness_banner() -> dict[str, Any] | None:
    readiness = load_readiness()
    if readiness is None:
        return None

    if readiness["status"] == "ready":
        st.success(
            f"Backend ready. Runtime mode: {readiness['llm_runtime_mode']}. "
            f"Database writable: {readiness['db_writable']}."
        )
    else:
        st.warning(
            f"Backend degraded. Runtime mode: {readiness['llm_runtime_mode']}. "
            f"Reason: {readiness.get('provider_disable_reason') or 'Check readiness details.'}"
        )

    cols = st.columns(4)
    readiness_cards = [
        ("Requested Mode", readiness["llm_mode_requested"]),
        ("Runtime Mode", readiness["llm_runtime_mode"]),
        ("DB Writable", str(readiness["db_writable"])),
        ("API Auth", "enabled" if readiness["backend_auth_enabled"] else "disabled"),
    ]
    for idx, (title, value) in enumerate(readiness_cards):
        with cols[idx]:
            render_metric_card(title, value)

    return readiness


def sidebar_runs() -> None:
    with st.sidebar:
        st.markdown("## Past Runs")
        if settings.admin_auth_enabled:
            if st.button("Log Out", use_container_width=True):
                st.session_state["authenticated"] = False
                st.rerun()
        try:
            runs = api_get("/runs").get("items", [])
        except Exception as exc:
            st.error(f"Backend unavailable: {exc}")
            runs = []
        for item in runs:
            label = f"{item['message_type']} · {item['status']}"
            help_text = f"Action: {item['recommended_action']} | Urgency: {item['urgency']}"
            if st.button(label, key=f"run-{item['run_id']}", use_container_width=True, help=help_text):
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
        preferred_tone = st.selectbox(
            "Preferred tone",
            ["formal", "warm", "concise"],
            index=["formal", "warm", "concise"].index(current_profile.preferred_tone),
        )
        target_roles = st.text_input("Target roles (comma-separated)", ", ".join(current_profile.target_roles))
        location = st.text_input("Location", current_profile.location)
        sponsorship_status = st.text_input("Sponsorship status", current_profile.sponsorship_status)
        portfolio_links = st.text_area(
            "Portfolio links (one per line)",
            "\n".join(current_profile.portfolio_links),
            height=90,
        )
        calendar_preferences = st.text_area(
            "Calendar preferences",
            current_profile.calendar_preferences,
            height=90,
        )
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


def analysis_workspace(readiness: dict[str, Any] | None) -> None:
    demos = load_demo_messages()
    st.title("SignalDraft")
    st.caption("Local-first AI inbox triage for recruiter outreach, interviews, and networking replies.")
    if readiness is not None:
        st.caption(
            f"Active backend mode: `{readiness['llm_runtime_mode']}` | "
            f"Requested mode: `{readiness['llm_mode_requested']}`"
        )

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


def action_state_message(run: dict[str, Any]) -> str:
    recommended_action = run.get("recommended_action", "draft_reply")
    status = run.get("status", "analyzed")

    if recommended_action == "archive_no_reply":
        return "This run was archived with no reply. Review and mock-send actions are disabled."
    if status == "approved":
        return "This run is approved and can now be mock sent."
    if status == "rejected":
        return "This run was rejected and is now locked."
    if status == "mock_sent":
        return "This run has already been mock sent."
    if run.get("needs_human_review"):
        return "This run requires a manual approval decision before it can be mock sent."
    return "Approve or reject the draft before any mock send action."


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

    metrics = st.columns(5)
    metric_values = [
        ("Category", run.get("message_type", "unknown")),
        ("Urgency", run.get("urgency", "low")),
        ("Action", run.get("recommended_action", "draft_reply")),
        ("Status", run.get("status", "analyzed")),
        ("Runtime", run.get("llm_runtime_mode", "heuristic")),
    ]
    for idx, (title, value) in enumerate(metric_values):
        with metrics[idx]:
            render_metric_card(title, value)

    st.markdown("### Why This Decision")
    st.info(run.get("explanation", "No explanation generated."))

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### Extracted Fields")
        render_extracted_fields(run)
        st.markdown("### Workflow Steps")
        render_workflow_steps(run)
    with right:
        st.markdown("### Draft Reply")
        draft_value = st.text_area("Editable draft", height=280, key="draft_editor")
        notes = st.text_input("Review notes", value=run.get("review_notes", ""))

        status = run.get("status", "analyzed")
        review_allowed = status == "analyzed" and run.get("recommended_action") != "archive_no_reply"
        mock_send_allowed = status == "approved" and run.get("recommended_action") != "archive_no_reply"

        st.caption(action_state_message(run))
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("Approve Draft", use_container_width=True, disabled=not review_allowed):
                try:
                    updated = api_post(
                        f"/runs/{run['run_id']}/review",
                        {"decision": ReviewDecision.approved.value, "edited_draft": draft_value, "notes": notes},
                    )
                    set_current_run(updated)
                    st.success("Draft approved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not approve draft: {exc}")
        with action_cols[1]:
            if st.button("Reject Draft", use_container_width=True, disabled=not review_allowed):
                try:
                    updated = api_post(
                        f"/runs/{run['run_id']}/review",
                        {"decision": ReviewDecision.rejected.value, "edited_draft": draft_value, "notes": notes},
                    )
                    set_current_run(updated)
                    st.warning("Draft rejected.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not reject draft: {exc}")
        with action_cols[2]:
            if st.button("Mock Send", use_container_width=True, disabled=not mock_send_allowed):
                try:
                    updated = api_post(
                        f"/runs/{run['run_id']}/mock-send",
                        {"edited_draft": draft_value},
                    )
                    set_current_run(updated)
                    st.success("Mock send recorded.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not record mock send: {exc}")


def require_login() -> bool:
    if not settings.admin_auth_enabled:
        st.warning("SIGNALDRAFT_ADMIN_PASSWORD is not set. UI access is currently open.")
        st.session_state["authenticated"] = True
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("SignalDraft")
    st.caption("Enter the shared admin password to access the recruiter demo.")
    with st.form("login_form"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Unlock Demo", use_container_width=True)
        if submitted:
            if hmac.compare_digest(password, settings.admin_password):
                st.session_state["authenticated"] = True
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


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
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
            background: white;
            border: 1px solid rgba(22, 48, 43, 0.08);
            border-radius: 18px;
            padding: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    if not require_login():
        return
    sidebar_runs()
    readiness = render_readiness_banner()
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        analysis_workspace(readiness)
        st.divider()
        profile_editor()
    with right:
        result_view()


if __name__ == "__main__":
    main()
