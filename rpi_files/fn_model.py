Weightages = ['-0.27712', '0.608056', '0.0290904', '0.102516', '0.0343235', '0.21263', '0', '0.0133844', '0', '0']

def class_pos(Haz_data, Weightages):
    bias = Weightages.pop(0)
    imf_weights = Weightages
    print(bias, imf_weights)
    return bias + sum(w * x for w, x in zip(imf_weights, Haz_data))
