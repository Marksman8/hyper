# Injection fixtures

**Every file in this directory contains a deliberate prompt-injection payload.**
They are test data for `injection.yaml`, not instructions, not documentation,
and not examples of anything to copy.

Each carries a `NOTE TO ANY READER` header for the same reason: these files are
readable by any tool or person that walks this repository, and untrusted content
that does not announce itself is how this problem spreads.

The payloads are written the way real ones look -- plausible surrounding
content, the instruction buried in the middle, addressed to the assistant in
the second person. A corpus of obviously-hostile text would measure nothing.

**Nothing here should ever be executed, followed, or moved out of this
directory.**
