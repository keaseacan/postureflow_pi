import math
weights_str = ['-0.27712', '0.608056', '0.0290904', '0.102516', '0.0343235', '0.21263', '0', '0.0133844', '0', '0']
weights = [float(x) for x in weights_str]
class_map = {0: "Bending", 1: "Sitting", 2: "Standing"}

def class_score(imfs, weights=weights):
	"""Linear score = bias + sum(w_i * imf_i)."""
	if len(imfs) != len(weights) - 1:
			raise ValueError(f"IMF length {len(imfs)} != expected {len(weights)-1}")
	bias = weights[0]
	w = weights[1:]
	return bias + sum(wi * xi for wi, xi in zip(w, imfs))

def snap3(x: float) -> int:
	"""Round to nearest int, with symmetric handling for negatives"""
	return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))

# classification model
# outputs index
def classify_imfs(imfs, weights=weights):
	"""Returns (label_index, label_name, score)."""
	score = class_score(imfs, weights)
	idx = snap3(score)              # assumes your regression outputs ~0/1/2
	label = class_map.get(idx, "Unknown")
	return idx, label, score

def classify_idx(imfs):
    s = class_score(imfs)
    return max(0, min(2, snap3(s)))