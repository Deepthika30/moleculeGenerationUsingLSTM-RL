import os
import sys

import torch
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# So "scripts.lstm_gen" and "scripts.property_predictors" work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.lstm_gen import LSTMGenerator
from scripts.property_predictors import predict_all

MODEL_PATH = "models/lstm_pretrained.pt"
DATA_PATH = "data/zinc_tokenized.pt"


# ---------- Load model + vocab once ----------
@st.cache_resource
def load_model_and_vocab():
    data = torch.load(DATA_PATH, map_location="cpu")
    stoi, itos = data["stoi"], data["itos"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMGenerator(vocab_size=len(stoi)).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, stoi, itos, device


def satisfies_constraints(
    props: dict,
    max_mw: float | None,
    max_logp: float | None,
    min_qed: float | None,
    max_toxic: float | None,
    must_contain: str,
) -> bool:
    """Same constraint logic as your CLI generate_by_constraints.py."""
    if max_mw is not None and props.get("MolecularWeight", 0.0) > max_mw:
        return False
    if max_logp is not None and props.get("LogP", 0.0) > max_logp:
        return False
    if min_qed is not None and props.get("QED", 0.0) < min_qed:
        return False
    if max_toxic is not None and props.get("Toxicity", 0.0) > max_toxic:
        return False
    if must_contain:
        comp = props.get("AtomComposition", {})
        if must_contain not in comp or comp[must_contain] == 0:
            return False
    return True


def generate_with_constraints_ui(
    target: int,
    max_mw: float | None,
    max_logp: float | None,
    min_qed: float | None,
    max_toxic: float | None,
    must_contain: str,
):
    """
    UI version of your generate_by_constraints.py loop.
    """
    model, stoi, itos, device = load_model_and_vocab()

    wanted = target
    max_attempts = target * 100  # exactly like your script
    rows = []
    attempts = 0
    hits = 0

    while hits < wanted and attempts < max_attempts:
        attempts += 1
        smi = model.sample(stoi, itos, device=device)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        props_out = predict_all(smi)
        # Support both dict and list-of-dicts, just in case
        if isinstance(props_out, list):
            if not props_out:
                continue
            props = props_out[0]
        elif isinstance(props_out, dict):
            props = props_out
        else:
            continue

        if not props.get("Valid", False):
            continue

        if not satisfies_constraints(props, max_mw, max_logp, min_qed, max_toxic, must_contain):
            continue

        props = props.copy()
        props["SMILES"] = smi
        rows.append(props)
        hits += 1

    return rows, attempts


# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="Molecular Generation – Constraints",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Molecular Generation with LSTM + RL (Constraint-based)")
st.write(
    "This UI mirrors your **generate_by_constraints.py** script:\n"
    "- LSTM generates SMILES\n"
    "- `predict_all` computes properties\n"
    "- Only molecules that satisfy your constraints are kept"
)

col_left, col_right = st.columns([1.4, 1])

with col_right:
    st.subheader("⚙️ Constraints (same as CLI)")

    target = st.number_input(
        "How many molecules do you want that satisfy constraints?",
        min_value=1, max_value=50, value=10, step=1,
    )

    st.caption("If you want to effectively disable a constraint, set it very high / low.")

    max_mw_val = st.number_input("Max molecular weight (e.g., 500)", value=500.0, step=10.0)
    max_logp_val = st.number_input("Max LogP (e.g., 5)", value=5.0, step=0.1)
    min_qed_val = st.number_input(
        "Min QED (0–1, e.g., 0.4)",
        value=0.4,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    max_toxic_val = st.number_input(
        "Max Toxicity (0–1, lower is better, e.g., 0.3)",
        value=0.2,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    must_contain = st.text_input(
        "Must contain element (e.g., C, N, Cl, Br)",
        value="C",
    ).strip()

    # Same semantics as CLI: None means "no constraint"
    max_mw = max_mw_val if max_mw_val > 0 else None
    max_logp = max_logp_val if max_logp_val > -10 else None
    min_qed = min_qed_val if min_qed_val > 0 else None
    max_toxic = max_toxic_val if max_toxic_val > 0 else None


with col_left:
    st.subheader("🚀 Generate molecules")

    if st.button("Generate", type="primary"):
        with st.spinner("Sampling and filtering molecules..."):
            rows, attempts = generate_with_constraints_ui(
                target=int(target),
                max_mw=max_mw,
                max_logp=max_logp,
                min_qed=min_qed,
                max_toxic=max_toxic,
                must_contain=must_contain,
            )

        if not rows:
            st.error(
                f"No molecules satisfied the constraints after {attempts} attempts.\n\n"
                "Try relaxing Max MW / Max LogP / Min QED / Max Toxicity."
            )
        else:
            st.success(
                f"Found **{len(rows)}** molecules satisfying constraints "
                f"(after {attempts} attempts)."
            )

            df = pd.DataFrame(rows)

            cols = [
                c for c in [
                    "SMILES", "MolecularWeight", "LogP", "QED",
                    "Solubility", "Rigidity", "Toxicity", "LipinskiPass"
                ]
                if c in df.columns
            ]
            if cols:
                st.dataframe(df[cols], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

            st.download_button(
                "📥 Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="constraint_filtered_molecules_ui.csv",
                mime="text/csv",
            )

            # Show molecule images (like your CLI RDKit grid)
            smiles_list = df["SMILES"].head(10).tolist()
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            mols = [m for m in mols if m is not None]
            if mols:
                img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(220, 220))
                st.image(img, caption="Sample generated molecules", use_container_width=False)
    else:
        st.info("Set constraints on the right and click **Generate**.")
