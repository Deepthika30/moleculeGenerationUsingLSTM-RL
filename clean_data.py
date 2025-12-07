
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
import re, torch
from torch.nn.utils.rnn import pad_sequence
from itertools import chain

df = pd.read_csv("data/raw.csv")
print("Total molecules in dataset:", len(df))

def clean_smiles(smiles_list):
    remover = SaltRemover.SaltRemover()
    valid = set()
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol = remover.StripMol(mol)
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                valid.add(canonical)
        except:
            continue
    return list(valid)

cleaned = clean_smiles(df['smiles'].tolist())
print("After cleaning:", len(cleaned))

allowed = {'C','H','N','O','F','S','Cl','Br','I','P'}
filtered = []
for s in cleaned:
    mol = Chem.MolFromSmiles(s)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    if all(a in allowed for a in atoms) and 10 <= len(s) <= 120:
        filtered.append(s)
print("After filtering:", len(filtered))

def tokenize_smiles(smi):
    regex = "(\[[^\[\]]{1,6}\])"
    tokens = []
    for token in re.split(regex, smi):
        if not token: continue
        if token.startswith('['):
            tokens.append(token)
        else:
            tokens += list(token)
    return ['<bos>'] + tokens + ['<eos>']

tokenized = [tokenize_smiles(s) for s in filtered]
print("Example tokenized:", tokenized[0][:25])

vocab = sorted(list(set(chain(*tokenized))))
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}
print("Vocabulary size:", len(vocab))

encoded = [torch.tensor([stoi[t] for t in seq]) for seq in tokenized]
padded = pad_sequence(encoded, batch_first=True, padding_value=stoi['<eos>'])

torch.save({"data": padded, "stoi": stoi, "itos": itos}, "data/zinc_tokenized.pt")
print("✅ Tokenized dataset saved as data/zinc_tokenized.pt")
