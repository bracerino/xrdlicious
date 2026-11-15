"""Small branding pieces shared by the apps: GitHub icon and supporter logos."""

import base64
from pathlib import Path

import streamlit as st

_IMAGES = Path(__file__).resolve().parent.parent / "images"

# GitHub mark (Octicons "mark-github", MIT license). Inherits the text color and
# size of the surrounding line, so it fits into any sentence.
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="currentColor" '
    'aria-hidden="true" style="vertical-align:-0.125em;margin-right:0.25em;">'
    '<path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17'
    '-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65'
    '-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64'
    '-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1'
    '-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2'
    '-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27'
    '.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21'
    '-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>'
)


def github_link_html(text="GitHub page",
                     url="https://github.com/bracerino/xrdlicious"):
    return (f'<a href="{url}" target="_blank" style="white-space:nowrap;">'
            f'{GITHUB_ICON}<b>{text}</b></a>')


def _data_uri(path):
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# (name, website, logo file, logo height in px)
SUPPORTERS = [
    ("FZU – Institute of Physics of the Czech Academy of Sciences",
     "https://www.fzu.cz/en", "fzu_logo.svg", 34),
    ("IMPLANT group", "https://implant.fs.cvut.cz/", "implant_logo.png", 44),
]


def render_supporters(container):
    logos = []
    for name, url, fname, height in SUPPORTERS:
        path = _IMAGES / fname
        if not path.exists():
            continue
        logos.append(
            f'<a href="{url}" target="_blank" title="{name}">'
            f'<img src="{_data_uri(path)}" alt="{name}" '
            f'style="height:{height}px;width:auto;"></a>')
    # Keyed so Streamlit tracks the block by key, not by position: otherwise a
    # rerun that adds sidebar content above it (e.g. Calculate XRD) briefly
    # shows the old and the new copy at once.
    box = container.container(key="sidebar_supporters")
    box.markdown("---")
    box.markdown(
        '<div style="font-size:0.85em;opacity:0.8;margin-bottom:6px;">'
        'Supported by <b>FZU</b> and the <b>IMPLANT group</b></div>'
        '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
        + "".join(logos) + '</div>',
        unsafe_allow_html=True,
    )
