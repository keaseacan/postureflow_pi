import math
weights_str = ['-0.307493', '0.212682', '0.266026', '0.100825', '0.0400341', '0.0983247', '0.155449', '0.0346157', '0.0540047', '0.0380388']
weights = [float(x) for x in weights_str]
class_map = {0: "Sitting", 1: "Standing"}

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