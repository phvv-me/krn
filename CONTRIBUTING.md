# Contributing

Use Python 3.14 and mainboard from this directory. Run `mb dep install`, then
`mb run lint`, `mb run test`, and `mb run ci`. Both ty and pyrefly check src,
tests, examples and experiments. The website is its own app in `apps/krn.dev`, run with mb from that folder.

Hardware tests require `KRN_HARDWARE=1`. The default suite uses captured outputs.
Changes to behavior include README, changelog, and documentation updates.
Do not commit, push, tag, or publish without the owner's instruction.
