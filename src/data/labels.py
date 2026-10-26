"""Canonical issue_area category names (Supreme Court Database codes 1-13,
the classes that survive filtering in preprocessing.load_corpus — see
README's "A note on the paper's own Table 1" section for where these codes
come from)."""

AREA_NAMES = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment", 4: "Due Process",
    5: "Privacy", 6: "Attorneys", 7: "Unions", 8: "Economic Activity",
    9: "Judicial Power", 10: "Federalism", 11: "Interstate Relations", 12: "Federal Taxation",
    13: "Miscellaneous",
}


def area_name(code):
    return AREA_NAMES.get(int(code), f"Unknown ({code})")
