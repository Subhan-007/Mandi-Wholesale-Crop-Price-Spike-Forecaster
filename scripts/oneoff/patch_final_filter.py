import pathlib
p = pathlib.Path("04_clean_merge.py")
t = p.read_text(encoding="utf-8")
assert "final_cross_check" not in t, "already patched"

func = '''SPIKE_HIGH, SPIKE_LOW = 2.2, 0.45   # a single day this far from the other cities is a glitch


def final_cross_check(df, passes=2):
    """Cross-city check on the FINAL daily series (after backup fill), per crop. Rows are dropped, never refilled."""
    removed = []
    for _ in range(passes):
        keep = []
        for crop, g in df.groupby("crop"):
            pv = g.pivot_table(index="date", columns="city", values="price")
            others = pd.DataFrame({c: pv.drop(columns=c).median(axis=1) for c in pv.columns})
            cnt = pd.DataFrame({c: pv.drop(columns=c).notna().sum(axis=1) for c in pv.columns})
            r = pv / others
            r5 = r.rolling(5, center=True, min_periods=2).median()
            persist = ((r5 > CROSS_HIGH) | (r5 < CROSS_LOW)) & ((r > DAY_HIGH) | (r < DAY_LOW))
            spike = (r > SPIKE_HIGH) | (r < SPIKE_LOW)
            bad = ((persist | spike) & (cnt >= 3)).stack()
            idx = bad[bad].index
            flag = g.set_index(["date", "city"]).index.isin(idx)
            removed.append(g[flag])
            keep.append(g[~flag])
        df = pd.concat(keep, ignore_index=True)
    return df, pd.concat(removed, ignore_index=True)


'''
t = t.replace("def main():", func + "def main():", 1)

old = 'df = pd.concat(frames, ignore_index=True)\n    say(f"Days filled from a BACKUP'
new = ('df = pd.concat(frames, ignore_index=True)\n'
       '    df, r_final = final_cross_check(df)\n'
       '    r_final.to_csv(PROCESSED_DIR / "final_cross_removed.csv", index=False)\n'
       '    say(f"Final cross-city check removed {len(r_final)} rows (incl. backup-filled days): final_cross_removed.csv")\n'
       '    say(f"Days filled from a BACKUP')
assert old in t, "anchor not found"
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("patched")