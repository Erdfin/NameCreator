from random import randint, choices

consonants = "TNSRHDLCMFWGPBVKXQJZ".lower()
consonants_w = [
    0.1519498722,
    0.1160304504,
    0.1048909409,
    0.1005578916,
    0.09889062944,
    0.07213198853,
    0.06644314361,
    0.04528173982,
    0.0436144777,
    0.03847527963,
    0.03498502212,
    0.03383076373,
    0.03037714935,
    0.02487152005,
    0.01849561657,
    0.01151510155,
    0.002885645972,
    0.001877960077,
    0.001722226803,
    0.001172579951
]

vowels = "aeiouy"
vowels_w = [
    0.2024828416,
    0.2995816357,
    0.182084165,
    0.1914495092,
    0.07172349676,
    0.0526783517
]

if __name__ == "__main__":
    weighted = []
    for i in range(150):
        syl = []
        for i in range(3):
            if randint(0, 1):
                syl.append("".join(choices(consonants, weights=consonants_w)))
                syl.append("".join(choices(vowels, weights=vowels_w)))
            else:
                syl.append("".join(choices(vowels, weights=vowels_w)))
                syl.append("".join(choices(consonants, weights=consonants_w)))
        weighted.append("".join(syl))

    for v1, v2, v3 in zip(weighted[:50], weighted[50:100], weighted[100:]):
        print(v1, "         ", v2, "         ", v3)