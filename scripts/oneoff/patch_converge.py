import pathlib
p = pathlib.Path("04_clean_merge.py")
t = p.read_text(encoding="utf-8")
assert "n_before" not in t, "already patched"
t = t.replace("def final_cross_check(df, passes=2):", "def final_cross_check(df, passes=8):", 1)
old = "        df = pd.concat(keep, ignore_index=True)\n    return df, pd.concat(removed, ignore_index=True)"
new = ("        n_before = len(df)\n"
       "        df = pd.concat(keep, ignore_index=True)\n"
       "        if len(df) == n_before:\n"
       "            break\n"
       "    return df, pd.concat(removed, ignore_index=True)")
assert old in t, "anchor not found"
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("patched")