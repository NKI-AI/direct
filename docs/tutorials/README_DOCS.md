# Documentation & Development Guidelines for DIRECT (for contributors)

This project requires **docstrings everywhere**:

* Every **module**, **class**, **function**, **method**, and even **helpers** must have a docstring.
* No exceptions: private helpers (`_foo`) and dunders (`__len__`, `__iter__`, etc.) must also be documented.

The Makefile provides commands to generate, clean, view, and serve documentation automatically.

Documentation is built automatically with **Sphinx** and deployed at:
**[https://docs.aiforoncology.nl/direct](https://docs.aiforoncology.nl/direct)**

---

## Docstring Rules

* Use **triple double quotes** (`"""`).
* Begin with a **one-line summary**.
* Add a longer description if needed.
* Document all arguments under `Args:`.
* Document return values under `Returns:`.
* Document raised exceptions under `Raises:`.
* Use double backticks for inline code or arguments written as:
  ```text
  ``like this``
  ```
  or
  ```text
  Args:
      param: Example description... Default is ``like this``.
  ```
* Do **not repeat types** in docstrings (types are already in annotations).
* Use reST roles (`:class:`, `:meth:`, `:attr:`) for cross-references.
* Use `.. math::` blocks for formulas.
* Include minimal runnable **Examples** with `>>>` when they add signal.
* Keep lines ≤ 120 characters.
* When citing external work, add a **References** section at the end of the docstring.
  Use Sphinx/reST citations with numbered markers.
  - In the docstring body, refer to a reference with `[ # ]_` (note the underscore).
  - At the end of the docstring, define the references:

  ```text
  References:
      .. [#] Reference 1 description
             Reference 1 description continuation
      .. [#] Reference 2 description
  ```

## Examples

### Module

```python
"""Loss functions for MRI reconstruction: SSIM and HFEN."""
```

### Class

```python
class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss.

    Computes the SSIM loss between predicted and target images.

    .. math::
        \\text{SSIM}(x, y) = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}
        {(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}

    Args:
        win_size: Window size for SSIM calculation. Default is ``7``.
        k1: Stability constant. Default is ``0.01``.
        k2: Stability constant. Default is ``0.03``.

    Examples:
        >>> loss = SSIMLoss()
        >>> pred = torch.rand(1, 1, 32, 32)
        >>> target = torch.rand(1, 1, 32, 32)
        >>> data_range = target.max().unsqueeze(0)
        >>> loss(pred, target, data_range)
        tensor(...)
    """
```

### Function

```python
def _shuffle_indices(indices: list[int]) -> list[int]:
    """Return a shuffled version of a list of indices.

    Args:
        indices: List of integer indices to shuffle.

    Returns:
        A new list containing the indices in random order.
    """
    return random.sample(indices, len(indices))
```

---

## Makefile Commands for Docs

The Makefile automates Sphinx documentation generation. Key commands:

* **`make docs`**
  Cleans old docs, generates API stubs via `sphinx-apidoc`, syncs the environment, and builds HTML.
  Warnings are logged to `docs/_build/warnings.log`.

* **`make clean-docs`**
  Removes built documentation and generated API stubs (`docs/_build/` and `docs/direct*.rst`).

* **`make viewdocs`**
  Opens the built docs in your browser.

* **`make uploaddocs`**
  Syncs docs to the remote server (`docs@aiforoncology.nl:/var/www/html/docs/direct`).
  Requires SSH access. Only use this if you have deploy access.

---

## Full Workflow

1. Write code → add **docstrings everywhere**.
2. Run `make docs` to rebuild API + HTML docs.
3. Run `make viewdocs` to preview.
4. Fix any warnings from `docs/_build/warnings.log`.
5. Push changes, then run `make uploaddocs` if you have deploy access.
