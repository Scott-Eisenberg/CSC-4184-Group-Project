import csv
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import pdfplumber

FOLDER = Path(__file__).resolve().parent
PDF_FILE = FOLDER / "Bioinformatics Pilot data - Sheet1.pdf"
OUTPUT_CSV = FOLDER / "pilot_extracted.csv"
SEQUENCES_CSV = FOLDER / "sequences.csv"

API_URL = "http://dgx03.ai.calpoly.edu:8002/v1/embeddings"

# Column order in the PDF table
TRAIT_COLUMNS = {
    6: "tnbc",
    7: "preeclampsia",
    8: "lupus",
}


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def infer_source_group(source_text, current_group):
    text = source_text.lower()

    if "ik" in text or "lupus risk" in text or "nature genetics" in text:
        return "lupus study"
    if "improved" in text or "polygenic risk" in text:
        return "breast cancer study"
    if "preeclampsia" in text or "maternal susceptibility" in text:
        return "preeclampsia study"

    # Some source text is split across rows/pages, so keep the last known group.
    return current_group


def extract_pilot_pdf():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"Put pilot.pdf in this folder: {FOLDER}"
        )

    records = []
    current_source_group = ""

    with pdfplumber.open(PDF_FILE) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 9:
                        continue

                    cells = [clean(cell) for cell in row]
                    rsid = cells[0]

                    # Skip headers and continuation rows that have no rsID.
                    if not re.fullmatch(r"rs\d+", rsid, flags=re.IGNORECASE):
                        continue

                    source_text = cells[1]
                    current_source_group = infer_source_group(
                        source_text, current_source_group
                    )

                    trait_values = {
                        label: cells[column_index]
                        for column_index, label in TRAIT_COLUMNS.items()
                    }
                    positive_traits = [
                        label for label, value in trait_values.items()
                        if value == "1"
                    ]

                    if len(positive_traits) != 1:
                        raise ValueError(
                            f"{rsid} should have exactly one trait label; "
                            f"found {trait_values}"
                        )

                    records.append({
                        "rsid": rsid.lower(),
                        "source_group": current_source_group,
                        "source_text_from_pdf": source_text,
                        "pathogenicity": cells[2],
                        "on_clinvar": cells[3],
                        "clinvar_link": cells[4],
                        "clinvar_pathogenicity": cells[5],
                        "tnbc": trait_values["tnbc"],
                        "preeclampsia": trait_values["preeclampsia"],
                        "lupus": trait_values["lupus"],
                        "label": positive_traits[0],
                    })

    if not records:
        raise ValueError("No variant rows were extracted from pilot.pdf.")

    fields = list(records[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    return records


def show_metrics(actual, predicted, class_names):
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    f1_scores = []
    recalls = []

    for label in class_names:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )
        recalls.append(recall)
        f1_scores.append(f1)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {sum(f1_scores) / len(f1_scores):.3f}")
    print(f"Balanced accuracy: {sum(recalls) / len(recalls):.3f}")


def get_embeddings(sequences):
    token = os.environ.get("EMBEDDINGS_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Set EMBEDDINGS_API_TOKEN in the VS Code terminal first."
        )

    embeddings = []

    for start in range(0, len(sequences), 8):
        batch = sequences[start:start + 8]
        body = json.dumps({
            "sequences": batch,
            "include_reverse_complement": False,
        }).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding API returned HTTP {error.code}: {details}"
            ) from error

        if "data" in result:
            embeddings.extend(item["embedding"] for item in result["data"])
        elif "embeddings" in result:
            embeddings.extend(result["embeddings"])
        else:
            raise RuntimeError(
                f"Unexpected API response keys: {list(result.keys())}"
            )

    return embeddings


def run_embedding_model(records):
    if not SEQUENCES_CSV.exists():
        print(
            "\nNo sequences.csv found, so embeddings were skipped. "
            "The PDF has rsIDs but no DNA sequences."
        )
        print(f"Extracted pilot data saved to: {OUTPUT_CSV}")
        return

    with open(SEQUENCES_CSV, newline="", encoding="utf-8") as file:
        sequence_rows = csv.DictReader(file)
        sequence_by_id = {
            row["rsid"].strip().lower(): row["sequence"].strip().upper()
            for row in sequence_rows
        }

    missing = [
        row["rsid"] for row in records
        if not sequence_by_id.get(row["rsid"])
    ]
    if missing:
        print("Missing sequences for:", ", ".join(missing))
        print("Add a real DNA sequence for every rsID to sequences.csv.")
        return

    sequences = [sequence_by_id[row["rsid"]] for row in records]
    for rsid, sequence in zip(
        [row["rsid"] for row in records], sequences
    ):
        if set(sequence) - set("ACGTN"):
            raise ValueError(f"{rsid} has characters outside A/C/G/T/N.")

    embeddings = get_embeddings(sequences)
    labels = [row["label"] for row in records]
    rsids = [row["rsid"] for row in records]

    dimensions = len(embeddings[0])
    with open(FOLDER / "embeddings.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["rsid", "label"] +
            [f"dimension_{i}" for i in range(dimensions)]
        )
        for rsid, label, vector in zip(rsids, labels, embeddings):
            writer.writerow([rsid, label] + vector)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=0,
        ),
    )
    predictions = cross_val_predict(
        model, embeddings, labels, cv=LeaveOneOut()
    )

    print(f"\nEmbedding vectors received: {len(embeddings)}")
    print(f"Dimensions per embedding: {dimensions}")
    print("Embedding + logistic regression, leave-one-out results:")
    show_metrics(labels, list(predictions), sorted(set(labels)))

    with open(FOLDER / "predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rsid", "actual", "predicted"])
        writer.writerows(zip(rsids, labels, predictions))


def main():
    records = extract_pilot_pdf()
    labels = [row["label"] for row in records]
    counts = Counter(labels)
    majority_label = counts.most_common(1)[0][0]
    baseline_predictions = [majority_label] * len(labels)

    print(f"Extracted {len(records)} rows from pilot.pdf.")
    print("Class counts:", dict(counts))
    print(f"Majority baseline predicts: {majority_label}")
    show_metrics(labels, baseline_predictions, sorted(set(labels)))
    print(f"Extracted table saved to: {OUTPUT_CSV}")

    run_embedding_model(records)


if __name__ == "__main__":
    main()