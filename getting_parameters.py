from heptapod_encode import batch_encode, save_encodings

encodings = batch_encode("logograms")
matrix, names = save_encodings(encodings, "phi_matrix.npy")
# matrix shape will be (49, 516) — your full dataset for ML