from ..contracts import Assessment
def synthetic_assessment(d):
    a = Assessment(summary=d.reason)
    if d.kind.startswith("contradiction"): a.contradictions = [d.reason]
    elif d.kind == "off_topic": a.off_topic = 0.9
    elif d.kind == "repetition": a.repetitive = True
    else: a.unsupported_claims = [d.reason]
    return a
