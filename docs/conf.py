# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import ast
import importlib
import inspect
import logging
import os
import sys
import warnings
from os.path import dirname, relpath

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath(".."))

curpath = os.path.dirname(__file__)
sys.path.append(os.path.join(curpath, "ext"))

# sphinx-autodoc-typehints still calls deprecated Sphinx APIs on Sphinx 9+.
try:
    from sphinx.deprecation import RemovedInSphinx10Warning

    warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)
except Exception:
    pass

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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

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
try:
    import myst_parser

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
except ImportError:
    pass

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
    except Exception:
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
        return None

    # Current module being documented (e.g., "direct.config")
    current_mod = name.rsplit(".", 1)[0] if "." in name else ""
    if not current_mod or not _is_pkg_init(current_mod):
        return None

    defining_mod = getattr(obj, "__module__", "") or ""
    if defining_mod and defining_mod != current_mod:
        # Sphinx 9+: options is _AutoDocumenterOptions (no item assignment).
        # Keep both spellings for older/newer autodoc option names.
        options.noindex = True
        options.no_index = True
    return None


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


def setup(app):
    """Register documentation hooks and the custom stylesheet.

    Args:
        app: Sphinx application object.
    """
    # This event lets you mutate autodoc options before rendering
    app.connect("autodoc-process-signature", noindex_pkg_init_reexports)
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
            try:
                return "duplicate object description" not in record.getMessage()
            except Exception:
                return True

    filt = _DropDuplicateObjectDescription()
    for name in list(logging.Logger.manager.loggerDict):
        if name == "sphinx" or name.startswith("sphinx."):
            logging.getLogger(name).addFilter(filt)
    logging.getLogger("sphinx").addFilter(filt)
