# Copyright (c) DIRECT Contributors
"""Sphinx roles that turn DOI and arXiv identifiers into hyperlinks.

Use the roles in documents as ``:doi:`10.1016/S0022-2836(05)80360-2``` or
``:arxiv:`1234.5678```. An explicit caption is also supported, for example
``:doi:`Basic local alignment search tool <10.1016/S0022-2836(05)80360-2>```.

Based on Sphinx ``extlinks`` and the original doilinks extension by Jon Lund Steffensen.
"""

from docutils import nodes, utils
from sphinx.util.nodes import split_explicit_title


def doi_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):  # noqa: B006
    """Create a hyperlink to a DOI resolver.

    Args:
        typ: Role name.
        rawtext: Raw role text including markup.
        text: Role contents.
        lineno: Line number of the role.
        inliner: Docutils inliner.
        options: Extra role options.
        content: Extra role content.

    Returns:
        A pair of ``(nodes, system_messages)`` as required by docutils roles.
    """
    text = utils.unescape(text)
    has_explicit_title, title, part = split_explicit_title(text)
    full_url = "https://doi.org/" + part
    if not has_explicit_title:
        title = "DOI:" + part
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []


def arxiv_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):  # noqa: B006
    """Create a hyperlink to an arXiv abstract.

    Args:
        typ: Role name.
        rawtext: Raw role text including markup.
        text: Role contents.
        lineno: Line number of the role.
        inliner: Docutils inliner.
        options: Extra role options.
        content: Extra role content.

    Returns:
        A pair of ``(nodes, system_messages)`` as required by docutils roles.
    """
    text = utils.unescape(text)
    has_explicit_title, title, part = split_explicit_title(text)
    full_url = "https://arxiv.org/abs/" + part
    if not has_explicit_title:
        title = "arXiv:" + part
    pnode = nodes.reference(title, title, internal=False, refuri=full_url)
    return [pnode], []


def setup_link_role(app):
    """Register the DOI and arXiv roles on a Sphinx application.

    Args:
        app: Sphinx application object.
    """
    app.add_role("doi", doi_role, override=True)
    app.add_role("DOI", doi_role, override=True)
    app.add_role("arXiv", arxiv_role, override=True)
    app.add_role("arxiv", arxiv_role, override=True)


def setup(app):
    """Register this extension with Sphinx.

    Args:
        app: Sphinx application object.

    Returns:
        Extension metadata.
    """
    app.connect("builder-inited", setup_link_role)
    return {"version": "0.1", "parallel_read_safe": True}
