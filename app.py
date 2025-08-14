import streamlit as st
import os
import sys
import importlib.util

# Configure page
st.set_page_config(
    page_title="DataSentience-AIML",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #5B3B8C; /* purple */
        color: #ffffff; /* ensure text is visible */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .domain-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6c757d;
    }
    /* Global overrides for common white sections across sub-apps */
    .input-section,
    .result-card,
    .sudoku-section,
    .upload-section,
    .model-info {
        background-color: #5B3B8C !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    .input-section h1, .input-section h2, .input-section h3,
    .result-card h1, .result-card h2, .result-card h3,
    .sudoku-section h1, .sudoku-section h2, .sudoku-section h3,
    .upload-section h1, .upload-section h2, .upload-section h3,
    .model-info h1, .model-info h2, .model-info h3 {
        color: #ffffff !important;
    }
    .input-section a, .result-card a, .sudoku-section a, .upload-section a, .model-info a {
        color: #FFD866 !important; /* readable link color on purple */
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌐 DataSentience-AIML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI/ML Solutions Across Multiple Domains</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Safely read text files with encoding fallbacks
def read_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            with open(path, "r", errors="replace") as f:
                return f.read()

# Dynamically discover all modules under src/*/* that have an app.py
def discover_modules(base_dir: str = "src"):
    domains_map = {}
    path_map = {}
    if not os.path.isdir(base_dir):
        return domains_map, path_map
    for domain in sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]):
        domain_path = os.path.join(base_dir, domain)
        modules = []
        for module in sorted([m for m in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, m))]):
            app_path = os.path.join(domain_path, module, "app.py")
            if os.path.exists(app_path):
                modules.append(module)
                path_map[(domain, module)] = app_path
        if modules:
            domains_map[domain] = modules
    return domains_map, path_map

# Build domains and module path mapping
domains, module_paths = discover_modules()
if not domains:
    st.warning("No Streamlit app.py modules found under 'src/'. Please add tools with an app.py.")

# Domain selection
selected_domain = st.sidebar.selectbox("Select Domain", list(domains.keys())) if domains else None

# Module selection based on domain
selected_module = (
    st.sidebar.selectbox("Select Module", domains[selected_domain])
    if selected_domain else None
)

# About section in sidebar (always visible)
st.sidebar.markdown("""
**About**

This application integrates various AI/ML solutions developed as part of the Social Summer of Code 2025 & GirlScript Summer of Code 2025 initiatives.

The project covers multiple domains including Healthcare, Finance, Agriculture, NLP, Safety, and more.

Select a domain and a specific module from the dropdowns above to explore different AI/ML solutions.
""")

# Main content area
if selected_domain and selected_module:
    st.markdown(f'<div class="domain-title">{selected_domain} » {selected_module}</div>', unsafe_allow_html=True)

# Function to load and run module
def load_module(module_path, module_name):
    try:
        # Add module directory to path
        module_dir = os.path.dirname(module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return False
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        st.error(f"Error loading module: {str(e)}")
        return False

# Try to load the selected module
if selected_domain and selected_module:
    module_path = module_paths.get((selected_domain, selected_module))
else:
    module_path = None

if module_path and os.path.exists(module_path):
    success = load_module(module_path, selected_module.replace(" ", "_"))
    if not success:
        st.warning(f"The module '{selected_module}' could not be loaded directly.")
        st.info("Here's a description of what this module does:")
        
        # Display module description
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Try to load README for the module
            readme_path = os.path.join("src", selected_domain, selected_module, "README.md")
            if os.path.exists(readme_path):
                readme_content = read_file_text(readme_path)
                st.markdown(readme_content)
            else:
                st.write(f"This module provides AI/ML solutions for {selected_module}.")
                st.write("Detailed documentation is not available for this module.")
            
            st.markdown('</div>', unsafe_allow_html=True)
elif selected_domain and selected_module:
    st.warning(f"The module '{selected_module}' does not have a Streamlit app.py file.")
    
    # Display module description
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Try to load README for the module
        readme_path = os.path.join("src", selected_domain, selected_module, "README.md")
        if os.path.exists(readme_path):
            readme_content = read_file_text(readme_path)
            st.markdown(readme_content)
        else:
            st.write(f"This module provides AI/ML solutions for {selected_module}.")
            st.write("Detailed documentation is not available for this module.")
        
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer"> 2025 DataSentience-AIML | Social Summer of Code 2025 & GirlScript Summer of Code 2025</div>', unsafe_allow_html=True)