import os
import yaml
import json
import random
from pathlib import Path

from accelerate import Accelerator
from torch import manual_seed
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
from datasets  import load_dataset
import os, json, pandas as pd, numpy as np
from leakpro import LeakPro
from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig, DataLoaderConfig, TrainingOutput, EvalOutput
from textInputHandler import TextInputHandler
import pickle
import torch
from torch import save

class C:                      # simple namespace
    MAX_LEN_CE       = 1024   # LM loss truncation
    CE_SUB_BS        = 16     # mini‑batch for CE pass
    FP16_CE          = True
    # ‑‑ neighbour settings
    NEI_K            = 0      # neighbours per passage (0 ⇒ vanilla CAMIA)
    NEI_WEIGHT       = 9      # weight in p‑value combiner (Edgington et al.)
    NEI_GAMMA        = 1.0    # multiplicative boost inside LogReg attack
    # signal hyper‑params
    CUT_TPRIMES      = {"T":None,"200":200,"300":300}
    CB_TAU_LIST      = [1,2,3]
    LZ_BIN_LIST      = [3,4,5]
    SLOPE_TPRIMES    = [600,800,1000]
    APEN_TPRIMES     = [600,800,1000]
    STORE_FULL_SEQ   = False  # saves RAM, switch on for diagnostics only
        # ── new elbow-signal hyper-params ────────────────────────────────
    ELBOW_SKIP_LEFT    = 25        # ignore the first/last ELBOW_SKIP tokens
    ELBOW_SKIP_RIGHT   = 0.25
    FLAT_SKIP_LEFT     = 0
    FLAT_SKIP_RIGHT    = 25
    ELBOW_STORE_I = False      # keep the raw index for later neighbour gen.
    FLAT_EPS = 1e-3
    
        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE}")

  

def fit_batch_size(model) -> int:
    """
    Shrinks C.CE_SUB_BS until one forward pass fits the free VRAM.
    No impact on the numeric results (only speed).
    """
    bs = C.CE_SUB_BS
    while True:
        try:
            dummy = torch.ones(bs, C.MAX_LEN_CE, dtype=torch.long, device=DEVICE)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=C.FP16_CE):
                model(dummy, attention_mask=torch.ones_like(dummy))
            break
        except torch.cuda.OutOfMemoryError:
            bs //= 2
            assert bs >= 1, "even BS=1 does not fit – lower MAX_LEN_CE"
            torch.cuda.empty_cache()
    if bs != C.CE_SUB_BS:
        print(f"Low-VRAM mode → CE_SUB_BS={bs}")
    C.CE_SUB_BS = bs
    return bs

def get_model_name(suite: str, size: str, deduped: bool = False) -> str:
    suite, size = suite.lower(), size.lower()
    if suite == "pythia":
        base = f"EleutherAI/pythia-{size}"
        return base + ("-deduped" if deduped else "")
    if suite == "gpt-neo":
        mapping = {"125m":"125M","1.3b":"1.3B","2.7b":"2.7B"}
        if size not in mapping:
            raise ValueError(f"Unsupported GPT‑Neo size: {size}")
        return f"EleutherAI/gpt-neo-{mapping[size]}"
    raise ValueError(f"Unknown suite: {suite}")  

# define the main function
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    manual_seed(SEED)
    
    # Load the config.yaml file
    script_dir = os.path.dirname(__file__)  # path to neurips.py
    config_path = 'train_config.yaml'
    audit_path = "audit.yaml"
    
    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)
    
    # ---------------- experiment grid ----------------
    GRID_NEI_K      = [25]#[0, 25, 100]
    GRID_NEI_WEIGHT = [72]          # matched 1-to-1 with GAMMA
    GRID_NEI_GAMMA  = [4]
    N_RUNS          = 1
    OUT_PARENT      = "elbow_neighbours_pile_cc_test"

    # ---------------- static model / data -----------
    domain_name  = train_config["data"]["dataset"]
    split_name   = "ngram_7_0.2"
    num_samples  = 1000 # number of audit samples (members + non-members)
    suite, size, dedup = "pythia", "70m", True
    model_tag    = f"{suite}{size}{'-dedup' if dedup else ''}".lower()

    #------------------------------------ load the Data  --------------------------------#
    dataset_name = f"{domain_name}_{split_name}"
    print(f"Loading {dataset_name}")

    # dataset
    hf_token = os.getenv("HF_TOKEN", "")
    hf_token = "<YOUR_HF_TOKEN>" # FILL IN YOUR TOKEN
    ds = load_dataset("iamgroot42/mimir", domain_name, split=split_name, token=hf_token)
    num_samples = min(num_samples, len(ds["member"]))
    all_members = ds["member"][:num_samples]
    all_nonmem  = ds["nonmember"][:num_samples]
    population_data = all_members + all_nonmem
    #------------------------------------ load the LM + Tokenizer  --------------------------------#
    model_name = get_model_name(suite,size,dedup)
    print("loading", model_name)
    acc   = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True)
    model = acc.prepare(model).eval()
    bs = fit_batch_size(model) # find the largest batchsize that fits VRAM
    
    #------------------------------------ Store data + Model --------------------------------#
    population_dataset = TextInputHandler.UserDataset(population_data, model_name, mode="raw")
        
    p_data = Path("data") / f"{dataset_name}.pkl"
    p_data.parent.mkdir(parents=True, exist_ok=True)   # makes 'data/' if missing
    with p_data.open("wb") as f:
        pickle.dump(population_dataset, f)
        print(f"Save data to {p_data}")

    save_dir = Path(train_config["run"]["log_dir"]) 
    save_dir.mkdir(parents=True, exist_ok=True) 
    acc.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
    print(f"Save model to {save_dir}")    
    
    n_train = len(all_members)
    n_test  = len(all_nonmem)
    train_indices = list(range(n_train))                      
    test_indices  = list(range(n_train, n_train + n_test)) 

    loss_config = LossConfig(name="crossentropyloss", params={})
    
    meta_data = MIAMetaDataSchema(
            init_params = {},
            optimizer = OptimizerConfig(name="", params=None),
            criterion = loss_config,
            data_loader = DataLoaderConfig(params={"batch_size":bs, "shuffle":False, "num_workers":0}),
            epochs = 1,
            train_indices = train_indices,
            test_indices = test_indices,
            num_train = len(train_indices),
            dataset_path = p_data.as_posix(),
            train_result = EvalOutput(accuracy=0.0, loss=0.0),
            test_result = EvalOutput(accuracy=0.0, loss=0.0),
        )
    p = Path(train_config["run"]["log_dir"]) / "model_metadata.pkl"
    with p.open("wb") as f:
        pickle.dump(meta_data, f)
    
    #------------------------------------ Run LeakPro  --------------------------------#
    leakpro = LeakPro(TextInputHandler, audit_path)
    result = leakpro.run_audit(create_pdf=False, use_optuna=False)
