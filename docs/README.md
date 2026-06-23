# Documentation bibliography workflow

The Sphinx docs cite works through `:cite:` roles (sphinxcontrib-bibtex,
author-year style). References are split across two files, both listed in
`conf.py`'s `bibtex_bibfiles`:

| File | Maintained by | Edit by hand? |
|---|---|---|
| `extract.bib` | **Generated** from the author's master library | **No** |
| `manual.bib` | Hand-maintained | Yes |

## Where references come from

Citations are sourced from the author's master BibTeX library,
`C:/S/TELOS/Biblio/uber-library.bib` (~7,100 entries, maintained with
[`archivum`](../../archivum_project)). That file is **read-only** from this
project — never edit it.

Read the Docs builds on Ubuntu runners with no access to `C:/S/...`, so
`bibtex_bibfiles` may only list repo-local files. `update_extract_bib.py`
bridges this: it scans every cite key used in `docs/**/*.rst`, pulls the
matching entries out of the master library, and writes them into the committed
`extract.bib`. Because the result is committed, RTD builds offline with every
reference present.

`manual.bib` holds the few references that intentionally are **not** in the
master library: a handful of academic works absent from it, plus software
citations (Python, SciPy, pandas, matplotlib, SLY, ...).

## After adding or changing a citation

1. Add the `:cite:` role to the `.rst` file, using the canonical
   `AuthorYYYY[a-z]` key from the master library.
2. Regenerate the extract on a machine with the master library available:

   ```
   uv run python docs/update_extract_bib.py
   ```

   (or run `docs/bib.bat`). The script exits non-zero and lists the offenders
   if any cited key is in neither the master library nor `manual.bib`, so
   broken citations are caught before commit.
3. Commit the regenerated `extract.bib`.

If a reference is missing from the master library, either add it there via
`archivum` (preferred — it then flows into `extract.bib` automatically on the
next run) or, for software / non-library works, add it to `manual.bib` by hand.

## Notes

- The master-library path defaults to `C:/S/TELOS/Biblio/uber-library.bib` and
  can be overridden with `--uber <path>` or the `UBER_LIBRARY` environment
  variable. The script only ever **reads** the library.
- Docs are not rebuilt in the normal iteration loop (the build is slow, 500+
  pages); the author rebuilds manually. Changing citations does require a
  rebuild to take effect.
