"""Voice: wake word, speech-to-text, text-to-speech, and the session loop.

Runs as its own process (`python -m aura voice`), talking to AURA Core over the
same authenticated HTTP API any other client uses. AURA Core has no idea voice
exists, and gains no dependency from it.
"""
