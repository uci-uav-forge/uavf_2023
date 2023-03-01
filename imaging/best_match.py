MATCH_THRESHOLD = 0.01
CONF_THRESHOLD = 0.2

def best_match(targets, letterConfidence, shapeConfidence):
    # targets: List[(letter, shape)]
    lC = dict(letterConfidence)
    sC = dict(shapeConfidence)
    r = max(targets, key = lambda t: lC[t[0]]*(sC[t[1]] if t[1] in sC else MATCH_THRESHOLD/2))



    return r