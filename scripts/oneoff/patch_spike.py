import pathlib
p = pathlib.Path("04_clean_merge.py")
t = p.read_text(encoding="utf-8")
old = "SPIKE_HIGH, SPIKE_LOW = 2.2, 0.45"
assert old in t, "anchor not found (already patched?)"
t = t.replace(old, "SPIKE_HIGH, SPIKE_LOW = 1.8, 0.55", 1)
p.write_text(t, encoding="utf-8")
print("patched")