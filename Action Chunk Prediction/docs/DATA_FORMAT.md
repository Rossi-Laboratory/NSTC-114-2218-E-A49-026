# DATA FORMAT
`preprocess_sequences.py` expects a JSON object:
{
  "actions": [[a1_1,...,a1_D], ..., [aT_1,...,aT_D]]
}
Outputs a `.pt` with tensors: `seqs [N,T,D]`, `boundaries [N,T]`.
