import math

Weightages = ['-0.27712', '0.608056', '0.0290904', '0.102516', '0.0343235', '0.21263', '0', '0.0133844', '0', '0']

def class_pos(Haz_data, Weightages):
    bias = Weightages.pop(0)
    imf_weights = Weightages
    print(bias, imf_weights)
    return bias + sum(w * x for w, x in zip(imf_weights, Haz_data))

def snap3(x: float) -> float:
    # round to nearest 0.5 with "half up", then clamp to [0, 1]
    y = math.floor(x * 2 + 0.5) / 2.0
    return min(1.0, max(0.0, y))

def class_num(num):
    class_map = {0: "Bending", 0.5: "Sitting", 1: "Standing"}
    return class_map.get(snap3(num), "Unknown")