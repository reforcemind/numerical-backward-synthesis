# Code standards (Ladder)

Standing agent: **Code Standards** - `.cursor/rules/code-standards.mdc`  
Ask: “Act as the code standards agent and clean this change.”

This repo is **C++20 primary**. Python appears only for optional scripts
(tt-metal ecosystem pins **3.10**). The same Ladder applies to both.

## Ladder - walk it before writing anything

| Step | Question | If yes |
|------|----------|--------|
| 1 | Does this need to exist? | If no -> skip (YAGNI) |
| 2 | Already in this codebase? | Reuse it |
| 3 | Stdlib / language facility? | Use it |
| 4 | Native platform feature? | Use it |
| 5 | Installed dependency? | Use it |
| 6 | Expressible in one line? | One line |
| 7 | Still needed? | Minimum that works |

## Rules

1. Never reimplement what a module here already provides - include/import it.
2. Use everything you add. Unused include, import, variable, argument, config key, or dependency: delete it.
3. Comments are short and detailed: record the decision a reader cannot infer from the line. A comment that restates its line is deleted.
4. No decorative comments. Banner rules (`// ----`), boxed headers, section dividers, and ASCII art are deleted on sight. If a file needs signposting it is too long - propose a split and wait for agreement.
5. No defensive branches for states that cannot occur.
6. Refactor inside existing files. Propose new structure and wait for agreement before creating it.

## C++ (primary)

- Standard: C++20 (`CMAKE_CXX_STANDARD 20`).
- Prefer existing `bw_syn` APIs (`bf16`, `contract`, `oracle`, `detail::scale_separated_product`, …).
- Prefer `<cmath>`, `<optional>`, `<span>`, `<string_view>`, standard algorithms over hand-rolled loops when clearer.
- Headers: include what you use; drop unused includes.
- Formatting: `.clang-format` (LLVM-based, 100 col). Run via `check_standards`.
- Warnings: `-Wall -Wextra -Wpedantic` (already on `bw_syn`).

## Python (rare)

- Newest API surface allowed by **3.10**: PEP 604 unions, `match`, dataclass `slots=True`, parenthesized context managers. Nothing from 3.11+.
- Tooling: `uvx ruff check .` and `uvx ruff format .`.
- Tests: `pytest` when Python tests exist (optional scripts may have none).

## Inspecting the repo

Read the file with the editor/tools. Do not write throwaway scripts or shell pipelines to discover what a file contains.

## End-of-turn checklist

```text
[ ] Ladder walked at each step?
[ ] Anything added that is now unused?
[ ] Anything reimplemented that already exists?
[ ] Any comment that only restates its line?
[ ] experiments/scripts/check_standards clean?
[ ] ctest --test-dir build green?
[ ] If Python touched: uvx ruff check . && uvx ruff format . clean?
[ ] If Python tests exist: pytest green?
```

Host one-liner:

```bash
# Unix
./experiments/scripts/check_standards.sh

# Windows PowerShell
./experiments/scripts/check_standards.ps1
```
