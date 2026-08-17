"""Telling a local user that a newer XRDlicious release is available.

A local copy is a snapshot of the repository at the moment it was cloned, so it
silently falls behind the deployed app. On a local run the newest release tag is
read from GitHub and, when it is ahead of the version baked into this copy, a
note is shown next to the release badge. Online deployments always run the
latest code, so the check is not made there (see ``IS_LOCAL`` in ``app.py``).

The lookup is cached and given a short timeout: no network, a rate-limited API
or a slow connection must never hold up the start of the app — in all those
cases the note is simply not shown.
"""

import re

import requests
import streamlit as st

# Version of this copy of the app, shown in the header of app.py / prdf.py.
APP_VERSION = "0.8.0"
APP_UPDATED = "August 17, 2026"

_LATEST_RELEASE_API = (
    "https://api.github.com/repos/bracerino/xrdlicious/releases/latest"
)
_REQUEST_TIMEOUT = 3       # seconds
_CACHE_TTL = 6 * 60 * 60   # ask GitHub at most a few times per day


def _version_tuple(text):
    """``"v0.7.3"`` -> ``(0, 7, 3)``; None when nothing numeric is found."""
    numbers = re.findall(r"\d+", text or "")
    if not numbers:
        return None
    return tuple(int(n) for n in numbers[:4])


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def latest_release_tag():
    """Newest release tag on GitHub, or None when it cannot be retrieved."""
    try:
        response = requests.get(
            _LATEST_RELEASE_API,
            timeout=_REQUEST_TIMEOUT,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status_code != 200:
            return None
        tag = response.json().get("tag_name")
        return tag.strip() if isinstance(tag, str) and tag.strip() else None
    except Exception:
        return None


def newer_release(current_version=APP_VERSION):
    """The released version that is newer than this copy, else None.

    A copy that is *ahead* of the newest release (a development version, or one
    released only after the tag) counts as up to date, so nothing is reported.
    """
    latest = latest_release_tag()
    current_parts, latest_parts = _version_tuple(current_version), _version_tuple(latest)
    if not current_parts or not latest_parts or latest_parts <= current_parts:
        return None
    return latest


def update_note_html(current_version=APP_VERSION):
    """HTML badge announcing a newer release; empty string when up to date."""
    latest = newer_release(current_version)
    if not latest:
        return ""
    return f"""
        <div style="
            display: inline-block;
            background-color: #fff7ed;
            border: 1px solid #f59e0b;
            border-radius: 4px;
            padding: 6px 12px;
            color: #7c2d12;
            font-size: 0.95rem;
            font-weight: 600;
        ">
            🔔 <span style="font-weight:800;">Update available:</span>
            {latest} &nbsp; | &nbsp;
            run <code>git pull</code> in the main XRDlicious folder to update
        </div>
    """
