**XRDlicious: Powder XRD/ND Patterns and PRDF Online Calculator**

An online, web-based tool for calculating powder X-ray and neutron diffraction (XRD and ND) patterns, as well as partial radial distribution functions, from crystal structures.
It features an integrated search interface for directly accessing and importing structures from the Materials Project (MP), AFLOW, and COD databases. Users can also interactively visualize uploaded structures, convert between conventional and primitive cells, and download the corresponding files in different formats (CIF, VASP, LMP, XYZ).

🔗 Try it here: [XRD/ND/PRDF Calculator](https://rdf-xrd-calculator.streamlit.app/)
🔗 [Tutorial how to use it HERE](https://implant.fs.cvut.cz/xrdlicious/)

For more computationally demanding calculations with more extensive data, please compile the code locally.
**How to compile and run the app:** (with activating the virtual environment to prevent possible conflicts between packages)
1) git clone https://github.com/bracerino/prdf-calculator-online.git
2) cd prdf-calculator-online/
3) python3 -m venv prdf_env
4) source prdf_env/bin/activate
5) pip install -r requirements.txt
6) streamlit run app.py
