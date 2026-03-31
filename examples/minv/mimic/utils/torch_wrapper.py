# utils/treegrad_wrapper.py
import os, joblib, yaml, numpy as np, pandas as pd, torch

class TorchWrapper:
    @classmethod
    def load_model(cls, model_path=None, **kwargs):
        """
        Accepts:
          - path to a file (e.g., ./target_tabular/model.ckpt),
          - path to a directory (e.g., ./target_tabular),
          - or None.
        Falls back to init_args.artifacts_dir or ./target_tabular.
        """
        artifacts_dir = None

        # Prefer explicit kwargs (from model_blueprint.init_args)
        artifacts_dir = kwargs.pop("artifacts_dir", None)

        # Infer from model_path when provided
        if artifacts_dir is None and model_path:
            if os.path.isfile(model_path):
                artifacts_dir = os.path.dirname(model_path)
            elif os.path.isdir(model_path):
                artifacts_dir = model_path

        # Final fallback
        if artifacts_dir is None:
            artifacts_dir = "./target"

        return cls(artifacts_dir=artifacts_dir, **kwargs)

    def __init__(self, artifacts_dir="./target"):
        self.artifacts_dir = artifacts_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = os.path.join(artifacts_dir, "config.yml")
        self.cfg = yaml.safe_load(open(cfg)) if os.path.exists(cfg) else {}

        meta = joblib.load(os.path.join(artifacts_dir, "custom_params.sav"))
        self.ohe = meta["ohe"]
        self.num_cols = meta["num_cols"]
        self.cat_cols = meta["cat_cols"]
        self.target_col = meta["target_col"]
        self.feature_order = meta["feature_order"]

        try:
            dm = joblib.load(os.path.join(artifacts_dir, "datamodule.sav"))
        except Exception:
            dm = {}
        self.classes_ = dm.get("classes_", None)
        self.n_classes_ = dm.get("n_classes_", None)

        bundle = joblib.load(os.path.join(artifacts_dir, "model.ckpt"))
        self.model = bundle["model"]
        self.eval()

    def __call__(self, entry: pd.DataFrame, requires_grad: bool = False) -> torch.Tensor:
        X_t = self._df_to_tensor(entry, requires_grad=requires_grad)
        try:
            out = self.model(X_t)
            if isinstance(out, torch.Tensor):
                return out.to(self.device)
        except Exception:
            pass
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X_t.detach().cpu().numpy())
            eps = 1e-12
            logits = np.log(np.clip(prob, eps, 1.0))
            return torch.tensor(logits, dtype=torch.float32, device=self.device)
        if hasattr(self.model, "decision_function"):
            dec = self.model.decision_function(X_t.detach().cpu().numpy())
            if dec.ndim == 1:
                dec = np.vstack([-dec, dec]).T
            return torch.tensor(dec, dtype=torch.float32, device=self.device)
        raise RuntimeError("Model lacks predict_proba/decision_function and torch-callable path.")

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        try:
            self.model.eval()
        except Exception:
            pass
        return self

    def _df_to_tensor(self, df: pd.DataFrame, requires_grad: bool) -> torch.Tensor:
        cols = [c for c in self.feature_order if c in df.columns]
        X_df = df[cols].copy()
        X_cat = self.ohe.transform(X_df[self.cat_cols]) if self.cat_cols else np.empty((len(X_df), 0))
        X_num = X_df[self.num_cols].to_numpy(dtype=float) if self.num_cols else np.empty((len(X_df), 0))
        X = np.hstack([X_cat, X_num]).astype("float32", copy=False)
        X_t = torch.from_numpy(X).to(self.device)
        X_t.requires_grad_(bool(requires_grad))
        return X_t
