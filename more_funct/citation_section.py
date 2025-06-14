import streamlit as st
from PIL import Image


def show_xrdlicious_roadmap():
    st.markdown("""
### Roadmap

* The XRDlicious will be regularly updated. The planned features are listed below. If you spot a bug or have a feature idea, please let us know at: lebedmi2@cvut.cz and we will gladly consider it.
-------------------------------------------------------------------------------------------------------------------

#### Code optimization 
* ⏳ Optimizing the code for better performance. ⏳Separate critical parameters (such as wavelength, new file, debye-waller factors) for diffraction patterns complete recalculations from non-critical (such as intensity scale or x-axis units) ✅ Optimized search in COD database.

#### Wavelength Input: Energy Specification
* ⏳ Allow direct input of X-ray energy (keV) for synchrotron measurements, converting to wavelength automatically.

#### Improved Database Search
* ✅ Add search by keywords, space groups, ids... in database queries.

#### Adding More Databases
* ⏳Potentially add additional databases such as NOMAD, or implement the OPTIMADE aggregator 

#### Expanded Correction Factors & Peak Shapes
* ⏳ Add more peak shape functions (Lorentzian, Pseudo-Voigt).
* ⏳ Introduce preferred orientation and basic absorption corrections.
* ⏳ Instrumental broadening - introduce Caglioti formula.
* ⏳ Calculate and apply peak shifts due to sample displacement error.

#### Enhanced Structure Visualization 
* ✅ Allow to change structure visualization style between Plotly and Py3Dmol
* ✅ Added option to change between perspective and orthogonal view

#### Enhanced Background Subtraction (Experimental Data)
* ⏳ Improve tools for background removal on uploaded experimental patterns.

#### Enhanced XRD Data Conversion
* ⏳ More accessible conversion interface - not hidden within the interactive plot.
* ⏳ Batch operations on multiple files at once (e.g., FDS/VDS, wavelength).
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
            image = Image.open("../images/Rb.png")
            st.image(image)
        with colh2:
            st.info("""
            From the **sidebar**, choose the calculation tool you'd like to use — **Structure Modification**, **Powder Diffraction Calculator**, **(P)RDF Calculator**, or **Interactive Data Plot**. Next, use the **sidebar** to **upload your crystal structure files** (**CIF**, **POSCAR**, **LMP**, or **XSF** formats) or your **two-column experimental data**.  
            If you don’t have crystal structure files, you can directly **add them using the search interface** for the **online databases**.
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
            "**Calculate powder XRD/ND patterns, (P)RDF, modify structures, and create point defects from crystal structures (CIF, LMP, POSCAR, XYZ), or perform peak matching and XRD data and file conversion.**\n\n"
            "Upload **structure files** (e.g., **CIF, LMP, POSCAR, XSF** format) and this tool will calculate either the "
            "**powder X-ray** or **neutron diffraction** (**XRD** or **ND**) patterns or **partial radial distribution function** (**PRDF**) for each **element combination**. Additionally, you can convert "
            "between primitive and conventional crystal structure representations, modify the structure, and introduce automatically interstitials, vacancies, or substitutes, downloading their outputs in CIF, POSCAR, LMP, or XYZ format. "
            "If **multiple files** are uploaded, the **PRDF** will be **averaged** for corresponding **element combinations** across the structures. For **XRD/ND patterns**, diffraction data from multiple structures are combined into a **single figure**."
            "There is also option to interactively plot and modify your two-columns data. In case of XRD data, you can convert between different wavelenghts, d-space, or q-space, and between fixed and automatic divergence slits. "
        )
        st.warning(
            "🪧 **Step 1**: 📁 Choose which tool to use from the sidebar.\n\n"
            "- **Structure Visualization** lets you view, convert (primitive ⇄ conventional), modify the structure (atomic elements, occupancies, lattice parameters) and download structures (**CIF, POSCAR, LMP, XYZ**). For creation of **supercells and point defects**, please visit [this site](https://xrdlicious-point-defects.streamlit.app/)\n\n"
            "- **Powder Diffraction** computes powder diffraction patterns on uploaded structures or shows **experimental data**.\n\n "
            "- **(P)RDF** calculates **partial and total RDF** for all element pairs on the uploaded structures.\n\n"
            "- **Peak Matching** allows users to upload their experimental powder XRD pattern and match peaks with structures from MP/AFLOW/COD databases. \n\n"
            "- **Interactive Data Plot** allows to plot two-column data and convert XRD data between wavelenghts, d-space and q-space. Additionally, it is possible to convert between fixed and automatic divergence slits.. \n\n"
            f"🪧 **Step 2**:  📁 Using the sidebar, upload your structure files or experimental patterns, or retrieve structures directly from MP, AFLOW, or COD crystal structure databases.."
            "Make sure the file format is supported (e.g., CIF, POSCAR, LMP, XYZ (with cell information))."
        )

        # from PIL import Image
        # image = Image.open("images/ts4.png")
        # st.image(image)
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
        - **XRDlicious, 2025** – for the interface.
        - **pymatgen** – for structure loading and powder diffraction pattern calculation, [S. P. Ong et al., pymatgen: A robust, open-source python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612006295).
        - **ASE (Atomic Simulation Environment)** – for structure loading, [A. H. Larsen et al., The Atomic Simulation Environment: A Python library for working with atoms, J. Phys.: Condens. Matter 29, 273002 (2017)](https://iopscience.iop.org/article/10.1088/1361-648X/aa680e).

        ---

        #### 🔁 **Using Calculated PRDF**
        - **XRDlicious, 2025** – for the interface.
        - **ASE** – for structure loading, [A. H. Larsen et al., The Atomic Simulation Environment: A Python library for working with atoms, J. Phys.: Condens. Matter 29, 273002 (2017)](https://iopscience.iop.org/article/10.1088/1361-648X/aa680e).
        - **pymatgen** – for structure loading, [S. P. Ong et al., pymatgen: A robust, open-source python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612006295).
        - **matminer** – for PRDF calculation, [L. Ward et al., matminer: An open-source toolkit for materials data mining, Comput. Mater. Sci. 152, 60 (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0927025618303252).

        ---

        #### 🏛️ **Using Structures from Databases**
        - **XRDlicious, 2025** – for the interface.
        - Cite the **relevant database**:
            - **Materials Project** [A. Jain et al., The Materials Project: A materials genome approach to accelerating materials innovation, APL Mater. 1, 011002 (2013)](https://pubs.aip.org/aip/apm/article/1/1/011002/119685/Commentary-The-Materials-Project-A-materials).
            - **AFLOW** [S. Curtarolo et al., AFLOW: An automatic framework for high-throughput materials discovery, Comput. Mater. Sci. 58, 218 (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612000717).,
            [M. Esters et al., aflow.org: A web ecosystem of databases, software and tools, Comput. Mater. Sci. 216, 111808 (2023)](https://www.sciencedirect.com/science/article/pii/S0927025622005195?casa_token=crrT7T_7vKoAAAAA:7UQbszQokpBT04i8kBqyN9JPXhaLf7ydlwuZen0taWZPXDx46zuYMPeaCJKeznY-BKKczMLzvw). 
            - **Crystallography Open Database (COD)** [S. Gražulis et al., Crystallography Open Database – an open-access collection of crystal structures, J. Appl. Crystallogr. 42, 726 (2009)](https://journals.iucr.org/j/issues/2009/04/00/kk5039/index.html).
        - **Important**: Always check the structure's original entry link in the database for any **associated publication** to cite.

        ---

        #### 📄 **Using XRD Data and File Conversion**
        - **XRDlicious, 2025**

        ---

        #### 🖼️ **Using Structure Visualizations**
        - **XRDlicious, 2025** – for the interface.
        - **pymatgen** – for structure loading, [S. P. Ong et al., pymatgen: A robust, open-source python library for materials analysis, Comput. Mater. Sci. 68, 314 (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0927025612006295).
        - **ASE** – for structure loading, [A. H. Larsen et al., The Atomic Simulation Environment: A Python library for working with atoms, J. Phys.: Condens. Matter 29, 273002 (2017)](https://iopscience.iop.org/article/10.1088/1361-648X/aa680e).
        - **Py3Dmol** – for 3D visualization, [N. Rego and D. Koes, 3Dmol. js: molecular visualization with WebGL, Bioinformatics 31, 1322 (2015)](https://academic.oup.com/bioinformatics/article/31/8/1322/213186).

        ---
        """)
