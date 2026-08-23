# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import ast
import importlib
import importlib.util
import inspect
import logging
import os
import shutil
import sys
import warnings
from contextlib import suppress
from os.path import dirname, relpath

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath(".."))

curpath = os.path.dirname(__file__)
sys.path.append(os.path.join(curpath, "ext"))

# sphinx-autodoc-typehints still calls deprecated Sphinx APIs on Sphinx 9+.
with suppress(ImportError):
    from sphinx.deprecation import RemovedInSphinx10Warning

    warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)

import direct

# -- Project information -----------------------------------------------------

project = "DIRECT"
copyright = "2025, AI for Oncology Research Group"
author = "DIRECT Contributors"

with open("../direct/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break

release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "myst_parser",
    "doi_role",
]

# sphinx-autodoc-typehints 3.6.x still uses Sphinx APIs removed in 9/10 and
# dumps hundreds of RemovedInSphinx10Warning + snippet parse errors. Prefer the
# built-in autodoc typehints settings below on Sphinx 9+.
import sphinx as _sphinx

_sphinx_major = int(_sphinx.__version__.split(".", 1)[0])
if _sphinx_major < 9:
    extensions.insert(7, "sphinx_autodoc_typehints")


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "_project_figures"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for autodoc -----------------------------------------------------

# This value is a list of autodoc directive flags that should be automatically
# applied to all autodoc directives.
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]

# This value controls the behavior of sphinx-build -W during importing modules.
# If False, warning messages during importing modules are ignored.
nitpicky = False

# -- Options for sphinx-apidoc -----------------------------------------------

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__, __call__",
    "undoc-members": False,
    "exclude-members": "__weakref__, model_config, _abc_impl",
    "private-members": True,  # Include _private methods
}

# Prefer full module paths over __init__.py imports
autodoc_mock_imports = []
autodoc_typehints = "description"

# -- Options for autosummary -------------------------------------------------
autosummary_generate = False

# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# --- How autodoc shows type hints (built-in Sphinx setting) ---
autodoc_typehints = "description"  # put types next to params/returns (not in signature)
autodoc_typehints_format = "short"

# --- Control how sphinx_autodoc_typehints emits return types ---
# (Option names from the extension docs.)
etypehints_use_rtype = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for MyST parser -------------------------------------------------
if importlib.util.find_spec("myst_parser") is not None:
    myst_enable_extensions = [
        "colon_fence",
        "deflist",
        "dollarmath",
        "html_admonition",
        "html_image",
        "replacements",
        "smartquotes",
        "substitution",
        "tasklist",
    ]
    # myst-parser, forcing to parse all html pages with mathjax
    # https://github.com/executablebooks/MyST-Parser/issues/394
    myst_update_mathjax = False

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Custom configuration ----------------------------------------------------
html_show_sourcelink = False
html_show_sphinx = False

# -- Options for search ------------------------------------------------------
html_search_language = "en"
html_search_options = {"type": "default"}

html_logo = "../logo/direct_logo_horizontal.svg"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_external_links": True,
}

# Suppress noisy tooling warnings that are not actionable docstring bugs.
# Duplicate package re-exports are handled by noindex_pkg_init_reexports below.
suppress_warnings = [
    "ref.footnote",
    "ref.ref",
    "docutils",
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
]

typehints_fully_qualified = False

# -- Options for module index -------------------------------------------------
# Remove alphabetic grouping since all modules start with 'direct.'
modindex_common_prefix = ["direct."]


def _is_pkg_init(modname: str) -> bool:
    try:
        m = importlib.import_module(modname)
    except (ImportError, OSError, ValueError):
        return False
    path = getattr(m, "__file__", "") or ""
    return os.path.basename(path).startswith("__init__.py")


def noindex_pkg_init_reexports(app, what, name, obj, options, signature, return_annotation):
    """For package ``__init__`` modules, mark members defined elsewhere as ``:noindex:``.

    Args:
        app: Sphinx application object.
        what: Object kind being documented.
        name: Fully-qualified object name.
        obj: Live object being documented.
        options: Autodoc options for the current object.
        signature: Object signature, if any.
        return_annotation: Return annotation, if any.

    Returns:
        Always ``None``. Mutates ``options`` in place when a re-export is detected.
    """
    # Only classes/exceptions are typically problematic, but you can drop this guard if you want.
    if what not in ("class", "exception"):
        return

    # Current module being documented (e.g., "direct.config")
    current_mod = name.rsplit(".", 1)[0] if "." in name else ""
    if not current_mod or not _is_pkg_init(current_mod):
        return

    defining_mod = getattr(obj, "__module__", "") or ""
    if defining_mod and defining_mod != current_mod:
        # Sphinx 9+: options is _AutoDocumenterOptions (no item assignment).
        # Keep both spellings for older/newer autodoc option names.
        options.noindex = True
        options.no_index = True


def linkcode_resolve(domain, info):
    """Return the GitHub URL corresponding to a Python object.

    Args:
        domain: Sphinx domain name. Only ``py`` is supported.
        info: Mapping with ``module`` and ``fullname`` keys from autodoc.

    Returns:
        A GitHub blob URL with a line range, or ``None`` when the object cannot
        be resolved to source.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, start_line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        stop_line = start_line + len(source) - 1
        linespec = f"#L{start_line}-L{stop_line}"

    fn = relpath(fn, start=dirname(direct.__file__))

    if "dev" in direct.__version__:
        return f"https://github.com/NKI-AI/direct/blob/main/direct/{fn}{linespec}"
    return f"https://github.com/NKI-AI/direct/blob/v{direct.__version__}/direct/{fn}{linespec}"


_DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_DOCS_DIR, ".."))
_PROJECT_README_PAGES = {
    "e2e_ads_recon": os.path.join(_REPO_DIR, "projects", "e2e_ads_recon"),
    "e2e_ads_recon_reg": os.path.join(_REPO_DIR, "projects", "e2e_ads_recon_reg"),
    "modulated_convolution": os.path.join(_REPO_DIR, "projects", "modulated_convolution"),
}


def copy_e2e_project_figures(app):
    """Copy e2e project figures into the Sphinx source tree.

    Project READMEs keep GitHub-relative ``figures/`` paths. Sphinx cannot
    collect images from outside ``docs/``, so copies live under
    ``_project_figures/<project>/`` during the build.

    Args:
        app: Sphinx application object.

    Returns:
        ``None``.
    """
    figures_root = os.path.join(_DOCS_DIR, "_project_figures")
    os.makedirs(figures_root, exist_ok=True)
    for name, project_dir in _PROJECT_README_PAGES.items():
        src_figures = os.path.join(project_dir, "figures")
        dest_figures = os.path.join(figures_root, name)
        if os.path.isdir(dest_figures):
            shutil.rmtree(dest_figures)
        if os.path.isdir(src_figures):
            shutil.copytree(src_figures, dest_figures)


def expand_e2e_project_includes(app, docname, source):
    """Inline e2e READMEs and rewrite ``figures/`` paths for Sphinx.

    The committed project pages are ``.. include::`` stubs, like the other
    project docs. ``source-read`` replaces that include with the README so
    figure paths can point at the in-tree copies under ``_project_figures``.

    Args:
        app: Sphinx application object.
        docname: Document name being read.
        source: Single-element list with the document source; mutated in place.

    Returns:
        ``None``.
    """
    project_dir = _PROJECT_README_PAGES.get(docname)
    if project_dir is None:
        return
    readme = os.path.join(project_dir, "README.rst")
    with open(readme, encoding="utf-8") as handle:
        text = handle.read()
    source[0] = text.replace(".. figure:: figures/", f".. figure:: _project_figures/{docname}/")


def copy_readme_banner(app):
    """Copy the repository README banner into the Sphinx static directory.

    Args:
        app: Sphinx application object.

    Returns:
        ``None``.
    """
    src = os.path.join(_REPO_DIR, "logo", "direct_banner.svg")
    dest = os.path.join(_DOCS_DIR, "_static", "direct_banner.svg")
    if os.path.isfile(src):
        shutil.copy2(src, dest)


def expand_root_readme(app, docname, source):
    """Inline the repository README on the docs homepage and fix local paths.

    The GitHub README uses ``logo/direct_banner.svg``. Sphinx serves a copy
    under ``_static/`` so the same raw HTML banner works in the docs.

    Args:
        app: Sphinx application object.
        docname: Document name being read.
        source: Single-element list with the document source; mutated in place.

    Returns:
        ``None``.
    """
    if docname != "index" or ".. include:: ../README.rst" not in source[0]:
        return
    readme_path = os.path.join(_REPO_DIR, "README.rst")
    with open(readme_path, encoding="utf-8") as handle:
        readme = handle.read()
    readme = readme.replace('src="logo/direct_banner.svg"', 'src="_static/direct_banner.svg"')
    readme = readme.replace(
        "`Apache 2.0 License <LICENSE>`__",
        "`Apache 2.0 License <https://github.com/NKI-AI/direct/blob/main/LICENSE>`__",
    )
    source[0] = source[0].replace(".. include:: ../README.rst", readme)


def setup(app):
    """Register documentation hooks and the custom stylesheet.

    Args:
        app: Sphinx application object.
    """
    # This event lets you mutate autodoc options before rendering
    app.connect("autodoc-process-signature", noindex_pkg_init_reexports)
    app.connect("builder-inited", copy_e2e_project_figures)
    app.connect("builder-inited", copy_readme_banner)
    app.connect("source-read", expand_e2e_project_includes)
    app.connect("source-read", expand_root_readme)
    app.add_css_file("custom.css")

    # Sphinx emits duplicate Field/Attributes docs without a suppressible type.
    class _DropDuplicateObjectDescription(logging.Filter):
        """Drop Sphinx 'duplicate object description' log records."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Return whether the log record should be kept.

            Args:
                record: Log record being filtered.

            Returns:
                ``False`` when the message is a duplicate-object warning.
            """
            return "duplicate object description" not in record.getMessage()

    filt = _DropDuplicateObjectDescription()
    for name in list(logging.Logger.manager.loggerDict):
        if name == "sphinx" or name.startswith("sphinx."):
            logging.getLogger(name).addFilter(filt)
    logging.getLogger("sphinx").addFilter(filt)
