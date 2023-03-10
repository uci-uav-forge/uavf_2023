MATCH_THRESHOLD = 0.01
CONF_THRESHOLD = 0.2

# approach here:
# color and letter are harder to get exactly right - 
# we should weight shape the most
# as it is the least noisy.
COLOR_WEIGHT = 0.4
LETTER_WEIGHT = 0.6

def best_match(targets, letterConfidence, shapeConfidence):
    # targets: List[(letter, shape)]
    lC = dict(letterConfidence)
    sC = dict(shapeConfidence)

    match_score = lambda t: LETTER_WEIGHT*lC[t[0]]+sC.get(t[1], MATCH_THRESHOLD/2)

    r = max(targets, key = match_score)


    return r, match_score(r)

def best_match_color(targets, 
                     letterColor, letterConfidence,
                     shapeColor, shapeConfidence):
    
    lC, sC = dict(letterConfidence), dict(shapeConfidence)
    
    match_score = lambda t: \
        COLOR_WEIGHT*letterColor[t[0]] + \
        LETTER_WEIGHT*lC[t[1]] + COLOR_WEIGHT*shapeColor[t[2]] + \
        sC.get(t[3], MATCH_THRESHOLD/2)

    result = max(targets, key = match_score)

    return result, match_score(result)