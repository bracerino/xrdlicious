import streamlit as st
from PIL import Image


def show_xrdlicious_roadmap():
    st.markdown("""
### Roadmap

* The XRDlicious will be regularly updated. The planned features are listed below. If you spot a bug or have a feature idea, please let us know at: lebedmi2@cvut.cz and we will gladly consider it.
-------------------------------------------------------------------------------------------------------------------

#### Code optimization 
* ⏳ Optimizing the code for better performance. 
* ✅ Separate critical parameters (such as wavelength, new file, debye-waller factors) for diffraction patterns complete recalculations from non-critical (such as intensity scale or x-axis units) 
* ✅ Optimized search in COD database.
* ✅ Repair visualization based on Wyckoff positions
#### Wavelength Input: Energy Specification
* ✅ Allow direct input of X-ray energy (keV) for synchrotron measurements, converting to wavelength automatically.

#### Improved Database Search / Adding More Databases
* ✅ Add search by keywords, space groups, ids... in database queries.
* ✅ Add search in MC3D database.
* ⏳Potentially add additional databases.

#### Expanded Correction Factors & Peak Shapes
* ✅ Add more peak shape functions (Lorentzian, Pseudo-Voigt).
* ⏳ Introduce preferred orientation and basic absorption corrections.
* ✅ Instrumental broadening - introduce Caglioti formula.
* ✅ Calculate and apply peak shifts due to sample displacement error.

#### Enhanced Structure Visualization 
* ✅ Allow to change structure visualization style between Plotly and Py3Dmol
* ✅ Added option to change between perspective and orthogonal view

#### Enhanced Background Subtraction (Experimental Data)
* ✅ Improve tools for background removal on uploaded experimental patterns.

#### Enhanced XRD Data Conversion
* ✅ More accessible conversion interface - not hidden within the interactive plot (available on https://xrd-convert.streamlit.app/, 🔄 X/Y-Axis Converter).
* ✅ Batch operations on multiple files at once (e.g., FDS/VDS, wavelength) (available on https://xrd-convert.streamlit.app/).
* ✅ Add conversion from PANalytical .xrdml and Rigaku .ras diffraction pattern file format to .xy file format 

#### Basic Peak Fitting (Experimental Data)
* ⏳ Fitting: Advanced goal for fitting profiles or full patterns to refine parameters.

#### Machine Learning 
* ⏳ Outlook for ML models for structure-properties correlations
""")

def first_run_note():
    if st.session_state["first_run_note"] == True:
        colh1, colh2 = st.columns([1, 3])
        with colh1:
            #image = Image.open("./images/Rb.png")
            image = Image.open("./images/cedule.png")
            st.image(image)

        with colh2:
            is_local = False
            try:
                host = st.context.headers.get("host", "")
                if "localhost" in host or "127.0.0.1" in host:
                    is_local = True
            except:
                pass

            if is_local:
                st.success(
                    "**Running locally.**\n\n"
                    "**Local limits (per structure):** up to **5,000,000** "
                    "estimated reflections in the limiting sphere — beyond "
                    "this the **Calculate** button is disabled to prevent "
                    "OOM crashes; **no peak-count truncation** (all peaks "
                    "are stored and displayed). Thresholds are editable "
                    "(`LOCAL_MAX_RECIP_POINTS`, `LOCAL_MAX_PEAKS` in "
                    "`more_funct/xrd_nd_section.py`)."
                )
            else:
                st.markdown(
                    """
<div style="
    background: #f8fbff;
    border-left: 4px solid #60a5fa;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0 10px 0;
    color: #1f2937;
    font-size: 0.95rem;
    line-height: 1.55;
">
<span style="font-size:1.05rem;">👋 <b>Hi there!</b></span><br>
This app currently runs on <b>free Streamlit hosting</b> with limited RAM (a hosting upgrade is planned). For heavier computations or a stable personal version, please consider to compile the app locally from <a href="https://github.com/bracerino/xrdlicious" target="_blank"><b>GitHub</b></a>.
<br><br>
<b>Online limits per structure:</b> up to <b>100,000</b> estimated reflections. Past that, the <b>Calculate</b> button is disabled to prevent memory overload. Additionally, the <b>top 1,500 peaks</b> by intensity are stored / displayed.
<br><br>
<b>Local limits:</b> <b>5,000,000</b> reflection cap and <b>no peak-count limit</b> (all peaks shown). Both thresholds are easy to tweak — see <code>LOCAL_MAX_RECIP_POINTS</code> and <code>LOCAL_MAX_PEAKS</code> in <code>more_funct/xrd_nd_section.py</code>.
</div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info("""
            Select a tool in the sidebar, then upload your structure files or data or import structures from online databases directly.
            """)

        st.session_state["first_run_note"] = False


def buttons_colors():
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #0099ff;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 0.5em 1em;
            border: none;
            border-radius: 5px;
            height: 3em;
            width: 100%;
        }
        div.stButton > button:active,
        div.stButton > button:focus {
            background-color: #0099ff !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        div[data-testid="stDataFrameContainer"] table td {
             font-size: 22px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def about_app():
    with st.expander("About the app.", icon="📖", expanded=True):
        st.info(
            "**XRDlicious** is an online toolbox for **powder XRD / neutron "
            "diffraction**, **structure inspection and editing**, and **XRD "
            "data conversion** — all from uploaded crystal structures "
            "(**CIF, POSCAR, LMP, XSF, XYZ**) or files fetched directly from "
            "the **Materials Project**, **AFLOW**, **COD**, or **MC3D** "
            "databases.\n\n"
            "Pick one or more tools from the **sidebar**, upload structures "
            "(or grab them from a database), then configure and run."
        )
        st.warning(
            "**🧰 Tools available in the sidebar**\n\n"
            "- **🔬 Structure Modification** — view in 3D, edit lattice "
            "parameters and atomic sites (element, fractional coordinates, "
            "occupancy), and download as **CIF / POSCAR / LAMMPS / XYZ**.\n\n"
            "- **💥 Powder Diffraction (XRD / ND)** — Rust-accelerated "
            "calculator (toggle the ⚡ icon to switch to pure pymatgen). "
            "Presets for **Cu, Mo, Co, Fe, Cr, Ag** including Kα1+Kα2(+Kβ) "
            "mixtures, plus custom λ or energy input. Optional **Debye-"
            "Waller** factors, **March-Dollase preferred-orientation "
            "correction**, **Scherrer + Caglioti broadening**, **sample-"
            "displacement** correction, five **peak profiles** (Delta, "
            "Gaussian, Lorentzian, Pseudo-Voigt, Pearson VII). Includes a "
            "**Reflection-Estimate tab** that blocks runaway cells before "
            "they OOM the server.\n\n"
            "- **📈 Interactive Data Plot** — plot two-column data; convert "
            "XRD between **wavelengths**, **d-space**, **q-space**, and "
            "between **fixed ↔ automatic divergence slits**.\n\n"
            "- **➡️ .xrdml ↔ .xy ↔ .ras Converter** — file-format conversion "
            "for common diffractometer outputs.\n\n"
            "- **↔️ Equivalent Planes** — list symmetry-equivalent (hkl) "
            "families for a given space group.\n\n"
            "**🔗 Companion apps** (opened separately)\n\n"
            "- **Calculate (P)RDF**: "
            "[Open App 🧩](https://prdf-xrdlicious.streamlit.app/) — "
            "partial / total radial distribution functions, averaged across "
            "multiple structures.\n"
            "- **Point defects, supercells & cell conversion**: "
            "[Open App 🌐](https://xrdlicious-point-defects.streamlit.app/) "
            "— vacancies, interstitials, substitutions, supercell builder, "
            "primitive ↔ conventional cell conversion.\n"
            "- **Convert between `.xrdml`, `.ras` and `.xy` formats or "
            "X/Y-axis**: "
            "[Open App 🧩](https://xrd-convert.streamlit.app/)\n"
            "- **Austenite-martensite crystallographic planes for NiTiHf**: "
            "[Open App 🌐](https://austenite-martensite.streamlit.app/)\n"
            "- **PRDF from LAMMPS / XYZ trajectories** (local, CLI): "
            "[Open Repo 🐙](https://github.com/bracerino/PRDF-CP2K-LAMMPS) "
            "— run locally; too heavy for the online server.\n\n"
            "**Tip:** when multiple structures are uploaded, XRD/ND "
            "patterns are overlaid in a single figure for direct comparison."
        )

        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()

def show_citation_section():
    with st.expander("How to Cite", icon="📚", expanded=True):
        st.markdown("""
        ### How to Cite

        Please cite the following sources based on the application usage:

        ---

        #### 🧪 **Using Calculated XRD Patterns**
        - **XRDlicious, 2025** – for the interface, [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://doi.org/10.1107/S1600576725005370)
        - **pymatgen** – for structure loading and powder diffraction pattern calculation, [S. P. Ong et al., pymatgen: A robust, open-source python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612006295).

        ---

        #### 🔁 **Using Calculated PRDF**
        - **XRDlicious, 2025** – for the interface, [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://doi.org/10.1107/S1600576725005370)
        - **pymatgen** – for structure loading, [S. P. Ong et al., pymatgen: A robust, open-source python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612006295).
        - **matminer** – for PRDF calculation, [L. Ward et al., matminer: An open-source toolkit for materials data mining, Comput. Mater. Sci. 152, 60 (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0927025618303252).

        ---

        #### 🏛️ **Using Structures from Databases**
        - **XRDlicious, 2025** – for the interface, [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://doi.org/10.1107/S1600576725005370)
        - Cite the **relevant database**:
            - **Materials Project** [A. Jain et al., The Materials Project: A materials genome approach to accelerating materials innovation, APL Mater. 1, 011002 (2013)](https://pubs.aip.org/aip/apm/article/1/1/011002/119685/Commentary-The-Materials-Project-A-materials).
            - **AFLOW** [S. Curtarolo et al., AFLOW: An automatic framework for high-throughput materials discovery, Comput. Mater. Sci. 58, 218 (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612000717).,
            [M. Esters et al., aflow.org: A web ecosystem of databases, software and tools, Comput. Mater. Sci. 216, 111808 (2023)](https://www.sciencedirect.com/science/article/pii/S0927025622005195?casa_token=crrT7T_7vKoAAAAA:7UQbszQokpBT04i8kBqyN9JPXhaLf7ydlwuZen0taWZPXDx46zuYMPeaCJKeznY-BKKczMLzvw). 
            - **Crystallography Open Database (COD)** [S. Gražulis et al., Crystallography Open Database – an open-access collection of crystal structures, J. Appl. Crystallogr. 42, 726 (2009)](https://journals.iucr.org/j/issues/2009/04/00/kk5039/index.html).
            - **Materials Cloud Three-Dimensional Structure Database (MC3D)** [S. P. Huber et al., MC3D: The materials cloud computational database of experimentally known stoichiometric inorganics, Digital Discovery 5, 1114 (2026)](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00415b).
        - **Important**: Always check the structure's original entry link in the database for any **associated publication** to cite.

        ---

        #### 📄 **Using XRD Data and File Conversion**
        - **XRDlicious, 2025**, [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.)

        ---

        #### 🖼️ **Using Structure Visualizations**
        - **XRDlicious, 2025** – for the interface, [Lebeda, Miroslav, et al. XRDlicious: an interactive web-based platform for online calculation of diffraction patterns and radial distribution functions from crystal structures. Applied Crystallography, 2025, 58.5.](https://doi.org/10.1107/S1600576725005370)
        - **pymatgen** – for structure loading, [S. P. Ong et al., pymatgen: A robust, open-source python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612006295).
        - **Py3Dmol** – for 3D visualization, [N. Rego and D. Koes, 3Dmol. js: molecular visualization with WebGL, Bioinformatics 31, 1322 (2015)](https://academic.oup.com/bioinformatics/article/31/8/1322/213186).

        ---
        """)
