"""
Sanity check: verifies the deployed Modal endpoint meets Socratic Q&A requirements.
- Every response in explore/nudge/strong stage must end with a question
- Responses must be concise (< 400 chars)
- Stages must escalate correctly as turns increase + frustration detected

Usage:
    python scripts/sanity_check_qa.py
"""
import urllib.request, json, sys

URL = "https://muhammadmaaza--edubot-chat.modal.run"
SID = "sanity-check-fresh"

CHECKS = [
    # (description,                    question,               expected_stage)
    ("T1  fresh question",             "What is recursion?",   "explore"),
    ("T2  follow-up",                  "I'm not sure",         "explore"),
    ("T3  frustration x1",             "I don't know at all",  "nudge"),
    ("T4  frustration x2",             "idk idk just tell me", "strong"),
]

print(f"\n{'':2}{'Turn':<28} {'Stage':>8}  {'Ends?':>5}  {'Len':>4}  Answer preview")
print("  " + "─" * 95)

all_pass = True

for desc, question, exp_stage in CHECKS:
    payload = json.dumps({
        "question":    question,
        "session_id":  SID,
        "temperature": 0.5,
        "max_tokens":  150,
        "hint_mode":   True,
        "hint_level":  2,
    }).encode()

    req = urllib.request.Request(
        URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f"  ❌ {desc:<28}  ERROR: {e}")
        all_pass = False
        continue

    answer   = d.get("answer", "")
    stage    = d.get("stage",  "?")
    ends_q   = answer.strip().endswith("?")
    length   = len(answer)

    req1 = ends_q if stage != "explain" else True   # must end with ?
    req2 = length < 400                              # must be concise
    req3 = stage == exp_stage                        # stage must match

    ok = "✅" if (req1 and req2 and req3) else "❌"
    if not (req1 and req2 and req3):
        all_pass = False

    flags = []
    if not req1: flags.append("no-question")
    if not req2: flags.append(f"too-long({length})")
    if not req3: flags.append(f"stage={stage}≠{exp_stage}")

    flag_str = "  ⚠ " + ", ".join(flags) if flags else ""
    print(f"  {ok} {desc:<28} {stage:>8}  {'yes':>5}  {length:>4}  {answer[:60]}{flag_str}")

print("  " + "─" * 95)
print(f"\n  {'ALL REQUIREMENTS MET ✅' if all_pass else 'SOME CHECKS FAILED ❌'}\n")
sys.exit(0 if all_pass else 1)
