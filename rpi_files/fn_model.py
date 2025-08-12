import math
Weightages = ['-0.27712', '0.608056', '0.0290904', '0.102516', '0.0343235', '0.21263', '0', '0.0133844', '0', '0']

def class_pos(Haz_data, Weightages):
    bias = Weightages.pop(0)
    imf_weights = Weightages
    print(bias, imf_weights)
    return bias + sum(w * x for w, x in zip(imf_weights, Haz_data))

def snap3(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))

def class_num(num):
    class_map = {0: "Bending", 1: "Sitting", 2: "Standing"}
    return class_map.get(class_pos(snap3(num), Weightages), "Unknown")