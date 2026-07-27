# `dev/deferred/` — parked, not closed

Plans deliberately **not** scheduled for `1.0.0b1`, kept intact so they can be
picked up after the beta. Parked is not the same as finished or abandoned, hence
the third folder:

| Folder | Meaning |
|---|---|
| `dev/` | live — being worked, or next up |
| `dev/deferred/` | parked past the beta cut; still wanted, no date |
| `dev/done/` | closed — shipped, `-REJECTED`, or `-SUPERSEDED` |

Each file here carries a banner saying **who deferred it, when, and what (if
anything) already ships in its place**. The matching `dev/TODO.md` entry keeps
the pointer, so nothing is parked silently.
