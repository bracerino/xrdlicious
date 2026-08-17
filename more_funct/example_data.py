"""The bundled SrTiO3 example — a structure plus its measured pattern.

A first-time visitor has nothing to look at until they find a structure file of
their own, so the welcome note offers a button that loads the two files in
``examples/`` as if they had been uploaded. The structure joins the same list as
the files coming from the database search, and the measured pattern is merged
into the experimental files by ``app.py``; from that point on the example is
handled by the ordinary code paths and can be removed like any other file.
"""

import io
from pathlib import Path

import streamlit as st

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_STRUCTURE = "SrTiO3.cif"
EXAMPLE_PATTERN = "SrTiO3_experiment.xy"
# Radiation and angular range the bundled pattern was measured with; the preset
# is one of PRESET_OPTIONS.
EXAMPLE_PRESET = "Co(Ka1+Ka2)"
EXAMPLE_TWO_THETA_MIN = 20.0
EXAMPLE_TWO_THETA_MAX = 125.0

_STRUCTURE_KEY = "example_structure_files"
_PATTERN_KEY = "example_pattern_files"
_ERROR_KEY = "example_load_error"


def _as_uploaded_file(path):
    """A file object indistinguishable from one of st.file_uploader's."""
    buffer = io.BytesIO(path.read_bytes())
    buffer.name = path.name
    return buffer


def load_example_data():
    """Register the example files. Meant as the button's on_click callback.

    Callbacks run before the script is executed again, so the files are already
    in session state by the time the page is built — a plain ``if st.button()``
    would not work here, because the welcome note that carries the button is not
    rendered any more on the run that handles the click.
    """
    st.session_state[_ERROR_KEY] = ""
    try:
        structure = _as_uploaded_file(EXAMPLES_DIR / EXAMPLE_STRUCTURE)
        pattern = _as_uploaded_file(EXAMPLES_DIR / EXAMPLE_PATTERN)
    except OSError as exc:
        st.session_state[_ERROR_KEY] = f"{exc}"
        return

    existing = st.session_state.get("uploaded_files") or []
    st.session_state["uploaded_files"] = (
        [f for f in existing if f.name != structure.name] + [structure]
    )
    st.session_state[_STRUCTURE_KEY] = [structure]
    st.session_state[_PATTERN_KEY] = [pattern]
    st.session_state["example_just_loaded"] = True

    # The point of the example is to show a pattern, so the calculation is
    # started as if 'Calculate XRD' had been pressed (the same two flags that
    # button sets).
    st.session_state["calc_xrd"] = True
    st.session_state["raw_patterns_cache_key"] = None

    # The bundled pattern was measured with a cobalt tube over a known angular
    # range, so both are set here — otherwise the calculated peaks would not
    # line up with the measured ones. The bookkeeping keys of the preset
    # selector are set as well: left alone, it would see this as a fresh preset
    # change and overwrite the range with its own suggestion.
    from more_funct.xrd_nd_section import (PRESET_WAVELENGTHS,
                                           set_two_theta_range)

    st.session_state["input_mode"] = "Preset"
    st.session_state["preset_choice"] = EXAMPLE_PRESET
    st.session_state["wavelength_value"] = PRESET_WAVELENGTHS[EXAMPLE_PRESET]
    st.session_state["_prev_preset"] = EXAMPLE_PRESET
    st.session_state["_prev_wavelength_value"] = float(
        PRESET_WAVELENGTHS[EXAMPLE_PRESET])
    set_two_theta_range(EXAMPLE_TWO_THETA_MIN, EXAMPLE_TWO_THETA_MAX)


def example_structure_files():
    """Example structure files currently loaded (empty list if none)."""
    return st.session_state.get(_STRUCTURE_KEY) or []


def example_pattern_files():
    """Example experimental patterns currently loaded (empty list if none)."""
    return st.session_state.get(_PATTERN_KEY) or []


def example_load_error():
    """Message of the last failed attempt, e.g. a missing examples/ folder."""
    return st.session_state.get(_ERROR_KEY) or ""


def remove_example_pattern(name):
    """Forget an example pattern.

    The example is re-attached to the uploaded experimental files on every run,
    so it cannot be taken away through the file uploader — dropping it here is
    what the removal button in the sidebar uses.
    """
    st.session_state[_PATTERN_KEY] = [
        f for f in example_pattern_files() if f.name != name
    ]


def merge_example_patterns(user_pattern_file):
    """The uploaded experimental files with the example appended.

    Anything the user uploaded wins: a file of the same name is not duplicated.
    """
    uploaded = list(user_pattern_file or [])
    names = {f.name for f in uploaded}
    return uploaded + [f for f in example_pattern_files() if f.name not in names]
