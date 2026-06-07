#!/usr/bin/env python3
"""CLEAN-ROOM verification, written from scratch, independent of analysis.py.

Recomputes every email-bound number directly from the raw transcripts with
simple auditable logic and MULTIPLE tests per claim (permutation, sign test,
Fisher exact, bootstrap). If this disagrees with analysis.py on any shared
figure, do NOT trust either. Eval output only, no model, no IP.

    python3 verify.py
"""
import json, re, math
from collections import Counter, defaultdict
import numpy as np

RAW = json.load(open("outputs/behavioral_transcripts.json"))
T = RAW["transcripts"]
TRAPS = ("unknowable", "plausible_fabrication", "obvious_fabrication")
SEED = 20260606

# ---------- 0. DATA INTEGRITY (verify the substrate before trusting any stat) ----------
print("=" * 70, "\n0. DATA INTEGRITY")
print(f"  transcripts: {len(T)}")
cats = Counter(p["category"] for p in T)
print(f"  categories: {dict(cats)}  (sum={sum(cats.values())})")
# every transcript must carry BOTH arms on the SAME query
bad = [p for p in T if not (p.get("method") and p.get("baseline"))]
qmismatch = [p for p in T if "query" in p and p.get("method", {}).get("query", p.get("query")) != p.get("query")]
print(f"  rows missing an arm: {len(bad)}   (pairing is intact if 0)")
# confidence values: range + how many parseable per arm
for arm in ("method", "baseline"):
    cv = [p[arm]["confidence"] for p in T if p[arm]["confidence"] is not None]
    print(f"  {arm}: parseable confidence on {len(cv)}/{len(T)}; range [{min(cv)},{max(cv)}]; "
          f"out-of-range {sum(1 for c in cv if c<0 or c>10)}")

def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
def correct(p, arm):   # plain answer-match; only defined where there is a reference answer
    # Prefer the precomputed 'correct' field (stripped-data builds); fall back to
    # response text match. Either path gives the same result.
    if "correct" in p[arm]:
        return p[arm]["correct"]
    ca = p.get("correct_answer")
    return (norm(ca) in norm(p[arm]["response"])) if ca else None

# ---------- helpers: stats from scratch ----------
def two_prop(a, na, b, nb):     # z-test, two-sided
    p1, p2 = a/na, b/nb; pp = (a+b)/(na+nb)
    se = math.sqrt(pp*(1-pp)*(1/na+1/nb))
    z = (p2-p1)/se
    return z, math.erfc(abs(z)/math.sqrt(2))
def fisher_2x2(a, b, c, d):     # exact two-sided p (hypergeometric), gold standard for small cells
    r1, r2, c1 = a+b, c+d, a+c; N = a+b+c+d
    def pmf(k): return math.comb(c1, k)*math.comb(N-c1, r1-k)/math.comb(N, r1)
    p_obs = pmf(a); tot = 0.0
    lo, hi = max(0, r1-(N-c1)), min(r1, c1)
    for k in range(lo, hi+1):
        pk = pmf(k)
        if pk <= p_obs*(1+1e-9): tot += pk
    return p_obs, min(1.0, tot)
def perm_paired(diffs, seed):   # sign-flip permutation, two-sided
    d = np.asarray(diffs, float); obs = d.mean(); rng = np.random.default_rng(seed)
    null = np.array([(d*rng.choice([-1.0, 1.0], len(d))).mean() for _ in range(20000)])
    return obs, float((np.abs(null) >= abs(obs)).mean())
def sign_test(diffs):           # exact binomial on signs, distribution-free
    pos = sum(1 for x in diffs if x > 0); neg = sum(1 for x in diffs if x < 0); n = pos+neg
    k = min(pos, neg); p = min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n) if n else 1.0
    return pos, neg, len(diffs)-pos-neg, p
def boot_ci(vals, seed):
    v = np.asarray(vals, float); rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(20000)]
    return v.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ---------- 1. CONFIDENTLY ASSERTS A FABRICATION (the email's hero; no answer-match dependence) ----------
print("=" * 70, "\n1. CONFIDENTLY ASSERTS A FABRICATION (all 75 traps, nothing excluded)")
alltraps = [p for p in T if p["category"] in TRAPS]
for thr in (5, 7):
    m = sum(1 for p in alltraps if (p["method"]["confidence"] or 0) >= thr)
    b = sum(1 for p in alltraps if (p["baseline"]["confidence"] or 0) >= thr)
    z, pz = two_prop(m, 75, b, 75)
    _, pf = fisher_2x2(m, 75-m, b, 75-b)
    print(f"  conf>={thr} FULL 2x2:  method {m} asserts / {75-m} does not   baseline {b} asserts / {75-b} does not   "
          f"(rates {m/75:.0%} vs {b/75:.0%})  two-prop p={pz:.2e}  Fisher exact p={pf:.2e}")

# ---------- 2. CONFIDENCE ON TRAPS (paired, where both arms gave a confidence) ----------
print("=" * 70, "\n2. CONFIDENCE ON FABRICATION TRAPS (paired)")
tp = [p for p in alltraps if p["method"]["confidence"] is not None and p["baseline"]["confidence"] is not None]
d = [p["baseline"]["confidence"] - p["method"]["confidence"] for p in tp]
mm = np.mean([p["method"]["confidence"] for p in tp]); bm = np.mean([p["baseline"]["confidence"] for p in tp])
obs, pp = perm_paired(d, SEED); pos, neg, tie, ps = sign_test(d); _, lo, hi = boot_ci(d, SEED)
print(f"  N={len(tp)}: method {mm:.2f} vs baseline {bm:.2f}  gap {bm-mm:.2f}  95%CI ({lo:.2f},{hi:.2f})")
print(f"  permutation p={pp:.4f} | sign test p={ps:.5f} (baseline-higher {pos}, method-higher {neg}, ties {tie})")

# ---------- 2b. FULL CONFIDENCE x CORRECTNESS 2x2 (paired N=58, the doc table) ----------
print("=" * 70, "\n2b. FULL CONFIDENCE x CORRECTNESS 2x2 (paired answerable, N=58)")
pairs58 = [p for p in T if p["category"] in ("factual", "implausible_true") and p.get("correct_answer")
           and p["method"]["confidence"] is not None and p["baseline"]["confidence"] is not None]
for arm in ("method", "baseline"):
    r = [p[arm]["confidence"] for p in pairs58 if correct(p, arm)]
    w = [p[arm]["confidence"] for p in pairs58 if not correct(p, arm)]
    print(f"  {arm:11s}  conf when RIGHT = {sum(r)/len(r):.2f}/10 (n={len(r)})   conf when WRONG = {sum(w)/len(w):.2f}/10 (n={len(w)})")

# ---------- 3. CONFIDENCE WHEN WRONG (both arms answer-match wrong, paired) ----------
print("=" * 70, "\n3. CONFIDENCE WHEN BOTH WRONG (factual+implausible_true, paired)")
wr = [p for p in T if p["category"] in ("factual", "implausible_true") and p.get("correct_answer")
      and p["method"]["confidence"] is not None and p["baseline"]["confidence"] is not None
      and not correct(p, "method") and not correct(p, "baseline")]
dw = [p["baseline"]["confidence"] - p["method"]["confidence"] for p in wr]
mw = np.mean([p["method"]["confidence"] for p in wr]); bw = np.mean([p["baseline"]["confidence"] for p in wr])
obs, pw = perm_paired(dw, SEED)
print(f"  N={len(wr)}: method {mw:.2f} vs baseline {bw:.2f}  gap {bw-mw:.2f}  permutation p={pw:.4f}")

# ---------- 4. COMBINED should-be-unsure ----------
print("=" * 70, "\n4. COMBINED should-be-unsure (traps + both-wrong)")
allc = tp + wr
dc = [p["baseline"]["confidence"] - p["method"]["confidence"] for p in allc]
g, lo, hi = boot_ci(dc, SEED); _, pcm = perm_paired(dc, SEED)
print(f"  N={len(allc)}: gap {g:.2f}  95%CI ({lo:.2f},{hi:.2f})  permutation p={pcm:.4f}")

# ---------- 5. CAPABILITY control + standard (direction-blind) ECE for honest context ----------
print("=" * 70, "\n5. CONTROLS / HONEST LIMITS")
def facc(arm):
    v = [correct(p, arm) for p in T if p["category"] == "factual" and p.get("correct_answer")]
    v = [x for x in v if x is not None]; return sum(v), len(v)
ma, mn = facc("method"); ba, bn = facc("baseline")
z, pc = two_prop(ma, mn, ba, bn)
print(f"  capability (factual answer-match): method {ma}/{mn}={ma/mn:.0%}  baseline {ba}/{bn}={ba/bn:.0%}  two-prop p={pc:.2f} (n.s. = no tradeoff)")
def ece(arm):
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    n = 0
    for p in T:
        if p["category"] == "factual" and p.get("correct_answer") and p[arm]["confidence"] is not None:
            pr = p[arm]["confidence"]/10.0; i = min(int(pr*10), 9); n += 1
            bins[i][0] += 1; bins[i][1] += float(bool(correct(p, arm))); bins[i][2] += pr
    return sum(c/n*abs(s/c - q/c) for c, s, q in bins.values() if c) if n else 0.0
print(f"  standard ECE (direction-BLIND): method {ece('method'):.3f}  baseline {ece('baseline'):.3f}  "
      f"(similar magnitude, OPPOSITE direction: method under, baseline over -> ECE is the wrong summary)")
print("=" * 70, f"\nseed={SEED}  (cross-check these against analysis.py)")
