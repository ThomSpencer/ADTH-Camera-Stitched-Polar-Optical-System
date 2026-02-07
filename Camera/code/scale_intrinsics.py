import os
import numpy as np

INPUT_FILE = "calibration/intrinsics_params.npz"
OUTPUT_FILE = "calibration/intrinsics_params_720p.npz"

# Old and new resolutions
OLD_SIZE = (640, 480)
NEW_SIZE = (1280, 720)


def main() -> None:
	if not os.path.isfile(INPUT_FILE):
		raise RuntimeError(f"Missing intrinsics file: {INPUT_FILE}")

	sx = NEW_SIZE[0] / OLD_SIZE[0]
	sy = NEW_SIZE[1] / OLD_SIZE[1]

	data = np.load(INPUT_FILE)
	out = {}
	for key in data.files:
		value = data[key]
		if key.endswith("_mtx"):
			mtx = value.copy()
			mtx[0, 0] *= sx
			mtx[1, 1] *= sy
			mtx[0, 2] *= sx
			mtx[1, 2] *= sy
			out[key] = mtx
		else:
			out[key] = value

	os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
	np.savez(OUTPUT_FILE, **out)
	print(f"Saved scaled intrinsics to {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
