"""Persisting the user's settings between local runs of the app.

Streamlit forgets everything when the server restarts, so a local user has to
re-enter the same diffraction / structure-modification options every time. The
helpers below dump the current widget state to a small JSON file next to
``app.py`` and re-apply it on the next start. Only offered for local runs — an
online deployment is shared between visitors, so one user's settings must not
leak into another's session.
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "xrdlicious_settings.json"

# Runtime state that must never be restored: uploaded data, database search
# results, caches, and the flags that decide whether a section has already been
# calculated. Restoring those would show stale results for files that are not
# loaded any more.
_SKIP_KEYS = {
    "calc_xrd", "calc_rdf", "ann_generated", "ann_mode",
    "uploaded_files", "full_structures", "full_structures_see_cod",
    "files_marked_for_removal", "first_run_note",
    "sidebar_uploader", "user_xrd", "latfit_upload",   # file uploaders
    "se_current_structure", "se_atoms", "se_atoms_loaded",
    "se_selected_file", "se_stored_orient",
    "parsed_exp_data", "permanent_exp_data", "bg_subtracted_data",
    "use_bg_subtracted", "active_bg_subtracted_file",
    "min_adjustments", "auto_stack_enabled",
    "raw_patterns_cache_key", "cif_space_groups",
    "display_mode", "selected_frame_idx",
}

_SKIP_PREFIXES = (
    "_user_settings",     # bookkeeping of this module itself
    "FormSubmitter:",     # internal Streamlit keys
    "$",
    "mp_", "cod_", "aflow_", "mc3d_",   # database search results / selections
    "latfit_peaktext",    # per-file peak lists
    "se_lat_",            # per-file lattice edits
    "_pageview",
)


# Streamlit refuses to let some widgets take their value from session state —
# buttons, download buttons, file uploaders, forms, camera/audio inputs. Seeding
# such a key raises StreamlitValueAssignmentNotAllowedError when the widget is
# created. Their values are momentary anyway (a click, an upload), so they are
# never stored, and anything left over in an older settings file is dropped by
# the guard installed in _install_widget_write_guard().
_TRIGGER_VALUE_TYPES = {"trigger_value", "string_trigger_value",
                        "json_trigger_value"}
_guard_installed = False


def _is_persistable(key):
    if not isinstance(key, str) or key in _SKIP_KEYS:
        return False
    return not key.startswith(_SKIP_PREFIXES)


def _trigger_widget_keys():
    """Keys of the buttons rendered in this run, read from Streamlit's state."""
    try:
        from streamlit.runtime.state import get_session_state

        state = get_session_state()
        inner = getattr(state, "_state", state)
        metadata = getattr(inner._new_widget_state, "widget_metadata", {}) or {}
        mapping = getattr(inner._key_id_mapper, "_key_id_mapping", {}) or {}
        return {
            key for key, widget_id in mapping.items()
            if getattr(metadata.get(widget_id), "value_type", None)
            in _TRIGGER_VALUE_TYPES
        }
    except Exception:
        return set()


def _install_widget_write_guard():
    """Drop a seeded value when its widget does not accept one.

    Every widget that forbids session-state writes goes through Streamlit's
    ``check_session_state_rules``, so patching that one function covers buttons,
    file uploaders, forms and anything added in a future release. Keys we seeded
    are removed just before the widget is built (instead of the whole section
    crashing), and every rejecting key is remembered so it is not saved again.
    """
    global _guard_installed
    if _guard_installed:
        return
    _guard_installed = True

    import sys
    from streamlit.elements.lib import policies

    original = policies.check_session_state_rules

    def guarded(default_value, key=None, writes_allowed=True):
        if key is not None and not writes_allowed:
            st.session_state.setdefault("_user_settings_no_write", set()).add(key)
            seeded = st.session_state.get("_user_settings_seeded")
            if seeded and key in seeded:
                seeded.discard(key)
                st.session_state.pop(key, None)
        return original(default_value, key, writes_allowed)

    policies.check_session_state_rules = guarded
    # A few element modules imported the function directly, so the patch above
    # does not reach them through the policies module.
    for module in list(sys.modules.values()):
        try:
            if getattr(module, "check_session_state_rules", None) is original:
                module.check_session_state_rules = guarded
        except Exception:
            continue


def collect_settings():
    """Every JSON-serialisable session value worth remembering."""
    settings = {}
    skip = _trigger_widget_keys() | set(
        st.session_state.get("_user_settings_no_write", set()))
    for key, value in st.session_state.items():
        if not _is_persistable(key) or key in skip:
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            # Structures, uploaded files, figures … cannot be stored.
            continue
        settings[key] = value
    return settings


def save_user_settings():
    """Write the current settings to disk. Returns (path, number of entries)."""
    settings = collect_settings()
    payload = {
        "format": 1,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings,
    }
    SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SETTINGS_FILE, len(settings)


def saved_settings_info():
    """(exists, saved_at) of the settings file."""
    if not SETTINGS_FILE.exists():
        return False, None
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return True, payload.get("saved_at")
    except Exception:
        return True, None


def delete_user_settings():
    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()
        st.session_state.pop("_user_settings_data", None)
        return True
    return False


def apply_saved_settings():
    """Seed the session with the saved settings.

    Must run at the top of the script, before any widget is created, otherwise
    Streamlit ignores the values. Keys that are already present are left alone,
    so anything the user changes during the session wins. It runs on every
    rerun because Streamlit drops the state of widgets that were not rendered
    (e.g. a calculation mode that is currently deselected), and those settings
    have to be seeded again the next time their section appears.
    """
    if "_user_settings_data" not in st.session_state:
        if not SETTINGS_FILE.exists():
            st.session_state["_user_settings_data"] = {}
        else:
            try:
                payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                st.session_state["_user_settings_data"] = payload.get("settings", {})
            except Exception:
                st.session_state["_user_settings_data"] = {}

    _install_widget_write_guard()
    seeded = st.session_state.setdefault("_user_settings_seeded", set())

    applied = 0
    for key, value in st.session_state["_user_settings_data"].items():
        if _is_persistable(key) and key not in st.session_state:
            st.session_state[key] = value
            seeded.add(key)
            applied += 1
    return applied
