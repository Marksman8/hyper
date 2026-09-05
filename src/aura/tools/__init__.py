"""Tools: the layer that lets AURA act, and the controls that make it safe.

Every call goes through registry -> validation -> workspace guard -> permission
-> execution -> audit. See docs/architecture/02-phase3-tools.md.
"""
