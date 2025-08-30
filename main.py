import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import subprocess
import kenlm


def levenshtein_distance(s1, s2):
    """Calculates the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def load_translation_dict(dict_path, encoding="utf-8-sig"):
    """Load QuocNgu to SinoNom translation dictionary"""
    df = pd.read_csv(dict_path, encoding=encoding)
    trans_dict = {}
    for _, row in df.iterrows():
        quoc_ngu = str(row["QuocNgu"]).strip().lower()
        sino_nom = str(row["SinoNom"]).strip()
        if quoc_ngu not in trans_dict:
            trans_dict[quoc_ngu] = set()
        trans_dict[quoc_ngu].add(sino_nom)
    return {k: list(v) for k, v in trans_dict.items()}


def load_similarity_dict(dict_path, encoding="utf-8-sig"):
    """Load SinoNom character similarity dictionary"""
    df = pd.read_csv(dict_path, encoding=encoding)
    similar_dict = {}
    for _, row in df.iterrows():
        input_char = str(row["Input Character"]).strip()
        similar_chars_str = str(row["Top 5 Similar Characters"]).strip()
        try:
            similar_chars = ast.literal_eval(similar_chars_str)
            similar_dict[input_char] = [input_char] + similar_chars
        except (ValueError, SyntaxError):
            similar_dict[input_char] = [input_char]
    return similar_dict


def is_compatible(sino_nom_char, quoc_ngu_word, trans_dict):
    if not sino_nom_char or not quoc_ngu_word:
        return False
    quoc_ngu_lower = quoc_ngu_word.strip().lower()
    possible_sino_chars = trans_dict.get(quoc_ngu_lower, [])
    return sino_nom_char in possible_sino_chars


def is_similarity_compatible(sino_nom_char, quoc_ngu_word, trans_dict, similar_dict):
    if not sino_nom_char or not quoc_ngu_word:
        return False, None
    quoc_ngu_lower = quoc_ngu_word.strip().lower()
    expected_sino_chars = trans_dict.get(quoc_ngu_lower, [])
    if not expected_sino_chars:
        return False, None
    similar_chars = similar_dict.get(sino_nom_char, [sino_nom_char])
    matches = list(set(expected_sino_chars) & set(similar_chars))
    if matches:
        return True, matches[0]
    return False, None


def levenshtein_align(sino_nom_text, quoc_ngu_text, trans_dict, similar_dict=None):
    """Simplified alignment function for error locating."""
    costs = {"match": 0, "mismatch": 1, "insertion": 1, "deletion": 1}
    sino_nom_chars = list(sino_nom_text.strip())
    quoc_ngu_words = quoc_ngu_text.strip().split()
    m, n = len(sino_nom_chars), len(quoc_ngu_words)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    backtrace = np.full((m + 1, n + 1), "", dtype=object)

    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            backtrace[i][0] = "U"
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            backtrace[0][j] = "L"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sino_nom_char = sino_nom_chars[i - 1]
            quoc_ngu_word = quoc_ngu_words[j - 1]
            match = is_compatible(sino_nom_char, quoc_ngu_word, trans_dict)
            subst_cost = costs["match"] if match else costs["mismatch"]
            options = [
                (dp[i - 1][j] + costs["deletion"], "U"),
                (dp[i][j - 1] + costs["insertion"], "L"),
                (dp[i - 1][j - 1] + subst_cost, "D"),
            ]
            dp[i][j], backtrace[i][j] = min(options)

    aligned_sino_nom, aligned_quoc_ngu, alignment_info = [], [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrace[i][j] == "D":
            sino_nom_char = sino_nom_chars[i - 1]
            quoc_ngu_word = quoc_ngu_words[j - 1]
            is_match = is_compatible(sino_nom_char, quoc_ngu_word, trans_dict)
            aligned_sino_nom.append(sino_nom_char)
            aligned_quoc_ngu.append(quoc_ngu_word)
            alignment_info.append(is_match)
            i -= 1
            j -= 1
        elif i > 0 and backtrace[i][j] == "U":
            aligned_sino_nom.append(sino_nom_chars[i - 1])
            aligned_quoc_ngu.append("_")
            alignment_info.append(False)
            i -= 1
        elif j > 0 and backtrace[i][j] == "L":
            aligned_sino_nom.append("_")
            aligned_quoc_ngu.append(quoc_ngu_words[j - 1])
            alignment_info.append(False)
            j -= 1

    aligned_sino_nom.reverse()
    aligned_quoc_ngu.reverse()
    alignment_info.reverse()
    return aligned_sino_nom, aligned_quoc_ngu, alignment_info


class NGramModel:
    def __init__(self, n, model_path="ngram.arpa", bin_path="ngram.binary"):
        self.n = n
        self.model_path = model_path
        self.bin_path = bin_path
        self.model = None

    def train(self, corpus_path):
        corpus_path = Path(corpus_path)  # ensure Path type
        print(f"Training {self.n}-gram KenLM model from {corpus_path}...")

        # Step 1: Extract clean training text
        clean_path = corpus_path.with_suffix(corpus_path.suffix + ".clean")
        with (
            corpus_path.open("r", encoding="utf-8") as f,
            clean_path.open("w", encoding="utf-8") as out,
        ):
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                text = " ".join(list(parts[1]))  # space between characters
                out.write(text + "\n")

        # Step 2: Train with KenLM (produces ARPA model)
        cmd = f"lmplz -o {self.n} < {clean_path} > {self.model_path}"
        subprocess.run(cmd, shell=True, check=True)

        # Step 3: Convert ARPA to binary (faster loading)
        subprocess.run(["build_binary", self.model_path, self.bin_path], check=True)

        print("Training complete. Binary model saved at", self.bin_path)

    def load(self):
        print(f"Loading KenLM model from {self.bin_path}...")
        self.model = kenlm.Model(self.bin_path)
        print("Model loaded.")

    def predict(self, context, candidates):
        """
        context: list of tokens (length <= n-1)
        candidates: list of possible next tokens
        """
        if not candidates:
            return None

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load().")

        context_str = " ".join(context)
        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:
            sentence = f"{context_str} {candidate}"
            score = self.model.score(sentence, bos=False, eos=False)
            if score > best_score or (
                score == best_score and candidate < best_candidate
            ):
                best_score = score
                best_candidate = candidate

        return best_candidate if best_candidate is not None else candidates[0]


def correct_text(
    sino_nom_text, quoc_ngu_text, mode, trans_dict, similar_dict, ngram_model=None
):
    aligned_sino, aligned_qn, alignment_info = levenshtein_align(
        sino_nom_text, quoc_ngu_text, trans_dict
    )
    corrected_sino = list(aligned_sino)

    for i, is_match in enumerate(alignment_info):
        if not is_match and corrected_sino[i] != "_":
            original_char = corrected_sino[i]
            quoc_ngu_word = aligned_qn[i]

            if mode in ["top-k", "combined"]:
                is_similar, replacement_char = is_similarity_compatible(
                    original_char, quoc_ngu_word, trans_dict, similar_dict
                )
                if is_similar:
                    corrected_sino[i] = replacement_char
                    continue

            if mode in ["contextual", "combined"]:
                context = ["<s>"] * (ngram_model.n - 1)
                if i > 0:
                    start = max(0, i - (ngram_model.n - 1))
                    context = corrected_sino[start:i]

                candidates = trans_dict.get(quoc_ngu_word.lower(), [])
                if candidates and ngram_model:
                    best_char = ngram_model.predict(context, sorted(candidates))
                    if best_char:
                        corrected_sino[i] = best_char

    return "".join(corrected_sino).replace("_", "")


def load_ground_truth(gt_path):
    gt_dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                gt_dict[parts[0]] = parts[1]
    return gt_dict


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Perform post-correction experiments based on alignment."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input text file (TSV: path\tnom\tquoc_ngu).",
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the corrected output."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["top-k", "contextual", "combined"],
        help="Correction mode.",
    )
    parser.add_argument(
        "--n-gram", type=int, default=3, help="Value of n for the n-gram model."
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[3]

    # --- Load resources ---
    dict_dir = project_root / "dict"
    trans_dict_path = dict_dir / "QuocNgu_SinoNom.csv"
    similar_dict_path = dict_dir / "SinoNom_Similar_Dic.csv"
    corpus_path = project_root.parent / "NomNaOCR" / "Patches" / "Train.txt"
    gt_path = project_root / "process" / "post-correction" / "assets" / "gt.txt"

    print("Loading resources...")
    if not all(
        [trans_dict_path.exists(), similar_dict_path.exists(), gt_path.exists()]
    ):
        print("Error: Resource files not found. Check paths.", file=sys.stderr)
        sys.exit(1)

    trans_dict = load_translation_dict(trans_dict_path)
    similar_dict = load_similarity_dict(similar_dict_path)
    gt_dict = load_ground_truth(gt_path)
    print(
        f"Loaded {len(trans_dict)} translation, {len(similar_dict)} similarity, and {len(gt_dict)} ground truth entries."
    )

    ngram_model = None
    if args.mode in ["contextual", "combined"]:
        if not corpus_path.exists():
            print(f"Error: Training corpus not found at {corpus_path}", file=sys.stderr)
            sys.exit(1)
        ngram_model = NGramModel(n=args.n_gram)
        ngram_model.train(corpus_path)
        ngram_model.load()  # <-- this step is required

    # --- Process input file ---
    print(f"Processing {args.input} with mode '{args.mode}'...")

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                df.append({"path": parts[0], "nom": parts[1], "quoc_ngu": parts[2]})

        df = pd.DataFrame(df)

    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_edit_distance = 0
    total_gt_length = 0

    for index, row in df.iterrows():
        path, nom_text, qn_text = (
            str(row["path"]),
            str(row["nom"]),
            str(row["quoc_ngu"]),
        )

        if not nom_text.strip() or not qn_text.strip():
            corrected_text = nom_text
        else:
            corrected_text = correct_text(
                nom_text, qn_text, args.mode, trans_dict, similar_dict, ngram_model
            )

        gt_text = gt_dict.get(path)
        line_cer_str = "N/A"
        if gt_text:
            edit_distance = levenshtein_distance(gt_text, corrected_text)
            gt_length = len(gt_text)
            if gt_length > 0:
                line_cer = edit_distance / gt_length
                total_edit_distance += edit_distance
                total_gt_length += gt_length
                line_cer_str = f"{line_cer:.4f}"

        results.append(
            f"{path}\t{nom_text}\t{qn_text}\t{corrected_text}\t{line_cer_str}\n"
        )
        if (index + 1) % 100 == 0:
            print(f"  ... processed {index + 1}/{len(df)} lines")

    # --- Write output & final CER ---
    with open(args.output, "w", encoding="utf-8") as f_out:
        f_out.writelines(results)

    print(f"\nCorrection complete. Output saved to {args.output}")

    if total_gt_length > 0:
        overall_cer = total_edit_distance / total_gt_length
        print(f"Overall CER: {overall_cer:.4%}")
    else:
        print("Could not calculate overall CER (no ground truth matches found).")


if __name__ == "__main__":
    main()
