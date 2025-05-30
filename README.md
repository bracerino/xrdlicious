# **XRDlicious: Powder XRD/ND Patterns and PRDF Online Calculator**

An online, web-based tool for calculating powder X-ray and neutron diffraction (XRD and ND) patterns, as well as partial radial distribution functions, from crystal structures.
It features an integrated search interface for directly accessing and importing structures from the Materials Project (MP), AFLOW, and COD databases. Users can also interactively visualize uploaded structures, convert between conventional and primitive cells, and download the corresponding files in different formats (CIF, VASP, LMP, XYZ).

🔗 Try it here: [XRD/ND/PRDF Calculator](https://rdf-xrd-calculator.streamlit.app/)
🔗 [Tutorial how to use it HERE](https://implant.fs.cvut.cz/xrdlicious/)

For more computationally demanding calculations with more extensive data, please compile the code locally.

# **How to compile and run the app:** 

**Prerequisities**: 
- Python 3.x (Tested 3.12)
- Console (For Windows, I recommend to do it in WSL2 (Windows Subsystem for Linux)
- Git (optional to download the code directly, otherwise go to the GitHub and download it manually,
(Optional): Install Git:
1) sudo apt update2)
2) sudo apt install git 

**Compile the app**
Into the console, write the following commands 
1) Download the XRDlicious code from GitHub (or download it manually without Git on the GitHub):
      git clone https://github.com/bracerino/xrdlicious.git

2) Go to the downloaded folder:
      cd xrdlicious/

3) Set a Python virtual environment to prevent possible conflicts between packages:
      python3 -m venv xrdlicious_env

4) Activate the Python virtual environment (when activating, make sure you are inside the xrdlicious folder):
      source xrdlicious_env/bin/activate
   
5) Install all the necessary Python packages:
      pip install -r requirements.txt

**Run the app**
6) Run the XRDlicious app (always before running it, do not forgot to activate its Python virtual environment (Step 4) :
      streamlit run app.py
