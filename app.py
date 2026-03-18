import streamlit as st
import torch
import numpy as np
import pandas as pd
import re
import requests
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
import py3Dmol
import streamlit.components.v1 as components
import torch.nn as nn

# ------------------------------------------------
# TOKENIZER
# ------------------------------------------------

SMILES_REGEX = r"Cl|Br|@@?|[B-DF-HK-NP-TV-Zb-df-hj-np-tv-z]|\d+|\(|\)|=|#|\+|\-|\\|\/|\."
regex = re.compile(SMILES_REGEX)

def tokenize(smiles):
    return regex.findall(smiles)

# ------------------------------------------------
# POSITIONAL ENCODING
# ------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=128):

        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):

        return x + self.pe[:,:x.size(1)]

# ------------------------------------------------
# MODEL
# ------------------------------------------------

class HybridTransformer(nn.Module):

    def __init__(self,vocab_size,num_tasks,sub_dim,prop_dim):

        super().__init__()

        d_model=128
        nhead=4
        num_layers=3
        dim_feedforward=256
        max_len=128

        self.embedding = nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model,max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(0.1)

        combined_dim = d_model + sub_dim + prop_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,num_tasks)
        )

    def forward(self,input_ids,attention_mask,sub_vec,prop_vec):

        x = self.embedding(input_ids)
        x = self.pos_encoder(x)

        key_padding_mask = attention_mask==0

        x = self.transformer(x,src_key_padding_mask=key_padding_mask)

        cls_rep = x[:,0,:]
        cls_rep = self.dropout(cls_rep)

        combined = torch.cat([cls_rep,sub_vec,prop_vec],dim=1)

        logits = self.mlp(combined)

        return logits

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

checkpoint = torch.load("hybrid_tox_model_full.pt",map_location="cpu")

vocab = checkpoint["vocab"]
toxicity_endpoints = checkpoint["toxicity_endpoints"]
sub_cols = checkpoint["sub_cols"]
prop_cols = checkpoint["prop_cols"]

model = HybridTransformer(
    vocab_size=len(vocab),
    num_tasks=len(toxicity_endpoints),
    sub_dim=len(sub_cols),
    prop_dim=len(prop_cols)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------------------------------------------------
# SUBSTRUCTURE SMARTS
# ------------------------------------------------

from rdkit import Chem

SUBSTRUCTURE_SMARTS = {
    "benzene":"c1ccccc1",
    "nitro":"[N+](=O)[O-]",
    "amide":"C(=O)N",
    "carboxylic_acid":"C(=O)[O;H,-]",
    "ester":"C(=O)O[C]",
    "alcohol":"[CX4][OH]",
    "amine":"[NX3]",
    "halogen":"[F,Cl,Br,I]",
    "ketone":"C(=O)[C]",
    "ether":"[OD2]([#6])[#6]",
}

compiled_substructures = {
    name: Chem.MolFromSmarts(smarts)
    for name,smarts in SUBSTRUCTURE_SMARTS.items()
}

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------

def encode_smiles(smiles):

    tokens = tokenize(smiles)
    tokens = ["<CLS>"] + tokens

    ids = [vocab.get(t,vocab["<UNK>"]) for t in tokens]

    max_len = 128

    if len(ids)<max_len:

        mask=[1]*len(ids)+[0]*(max_len-len(ids))
        ids += [vocab["<PAD>"]]*(max_len-len(ids))

    else:

        ids=ids[:max_len]
        mask=[1]*max_len

    return torch.tensor([ids]),torch.tensor([mask])


def name_to_smiles(drug_name):

    cleaned_name = drug_name.strip()

    if not cleaned_name:

        return None

    encoded_name = requests.utils.quote(cleaned_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES,ConnectivitySMILES/JSON"
    headers = {
        "User-Agent":"Streamlit-Toxicity-App/1.0"
    }

    try:

        response = requests.get(url,headers=headers,timeout=10)
        print("PubChem status:",response.status_code)
        print("Response preview:",response.text[:100])

        if response.status_code != 200:

            return None

        data = response.json()
        properties = data.get("PropertyTable",{}).get("Properties",[])

        if not properties:

            return None

        props = properties[0]
        smiles = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")

        return smiles

    except (requests.RequestException,ValueError,TypeError):

        return None


def compute_properties(mol_or_smiles):

    mol = mol_or_smiles if isinstance(mol_or_smiles,Chem.Mol) else Chem.MolFromSmiles(mol_or_smiles)

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    return [mw,logp,rings,hbd,hba]


def lipinski_rule(mw,logp,hbd,hba):

    results = {
        "MW < 500":mw < 500,
        "LogP < 5":logp < 5,
        "HBD <= 5":hbd <= 5,
        "HBA <= 10":hba <= 10
    }

    passed = sum(results.values())

    return results,passed


def interpret_risk(prob):

    if prob > 0.7:

        return "High Risk","red"

    elif prob > 0.4:

        return "Moderate Risk","orange"

    else:

        return "Low Risk","green"


def detect_substructures(mol_or_smiles):

    mol = mol_or_smiles if isinstance(mol_or_smiles,Chem.Mol) else Chem.MolFromSmiles(mol_or_smiles)

    detected=[]

    for name,pattern in compiled_substructures.items():

        if mol.HasSubstructMatch(pattern):

            detected.append(name)

    return detected

# ------------------------------------------------
# UI
# ------------------------------------------------

st.title("Molecular Toxicity Analysis Dashboard")

input_mode = st.radio("Select Input Type",["SMILES","Drug Name"])

if input_mode == "SMILES":

    user_input = st.text_input("Enter SMILES")

else:

    user_input = st.text_input("Enter Drug Name")

if st.button("Analyze Molecule"):

    smiles = None
    mol = None

    if input_mode == "SMILES":

        smiles = user_input.strip()

        if smiles:

            mol = Chem.MolFromSmiles(smiles)

        if mol is None:

            st.error("Invalid SMILES")

    else:

        drug_name = user_input.strip()

        if drug_name:

            smiles = name_to_smiles(drug_name)

            if smiles is not None:

                mol = Chem.MolFromSmiles(smiles)

        if mol is None:

            st.error("Drug not found")

        else:

            st.info(f"Converted '{drug_name}' -> {smiles}")

    if mol is not None:

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("2D Structure")
            st.image(Draw.MolToImage(mol,size=(350,350)))

        with col2:

            st.subheader("3D Structure")

            mol3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3d,AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol3d)

            mb = Chem.MolToMolBlock(mol3d)

            viewer = py3Dmol.view(width=350,height=350)
            viewer.addModel(mb,'mol')
            viewer.setStyle({'stick':{}})
            viewer.zoomTo()

            html = viewer._make_html()

            components.html(html,height=350)

        # ------------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------------

        input_ids,mask = encode_smiles(smiles)

        sub_vec = torch.zeros((1,len(sub_cols)))
        prop_vec = torch.tensor([compute_properties(mol)],dtype=torch.float)

        with torch.no_grad():

            logits = model(input_ids,mask,sub_vec,prop_vec)
            probs = torch.sigmoid(logits)[0].numpy()

        df = pd.DataFrame({
            "Endpoint":toxicity_endpoints,
            "Probability":probs
        })

        df = df.sort_values("Probability",ascending=False)

        prob = float(df.iloc[0]["Probability"])
        top_endpoint = df.iloc[0]["Endpoint"]
        label,color = interpret_risk(prob)

        st.subheader("Toxicity Assessment")

        if color == "red":

            st.error(f"{label} ({prob:.2f})")

        elif color == "orange":

            st.warning(f"{label} ({prob:.2f})")

        else:

            st.success(f"{label} ({prob:.2f})")

        st.caption(f"Highest predicted risk is for {top_endpoint}.")

        st.subheader("Toxicity Risk Ranking")

        st.bar_chart(df.set_index("Endpoint"))

        # ------------------------------------------------
        # SUBSTRUCTURES
        # ------------------------------------------------

        st.subheader("Detected Functional Groups")

        subs = detect_substructures(mol)

        if len(subs)==0:

            st.write("No common functional groups detected")

        else:

            for s in subs:

                st.success(s)

        # ------------------------------------------------
        # PROPERTIES
        # ------------------------------------------------

        st.subheader("Physicochemical Properties")

        mw,logp,rings,hbd,hba = compute_properties(mol)

        prop_df = pd.DataFrame({
            "Property":["Molecular Weight","LogP","Rings","HBD","HBA"],
            "Value":[mw,logp,rings,hbd,hba]
        })

        st.table(prop_df)

        st.subheader("Drug-likeness (Lipinski Rule)")

        results,passed = lipinski_rule(mw,logp,hbd,hba)

        for rule,status in results.items():

            if status:

                st.success(f"{rule} \u2714")

            else:

                st.error(f"{rule} \u2718")

        st.info(f"Passed {passed}/4 rules")
