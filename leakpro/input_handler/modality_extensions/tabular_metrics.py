"""TabularMetrics class for computing tabular metrics."""
import gower
import numpy as np
import pandas as pd
import torch
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import GMLogLikelihood
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

from leakpro.attacks.utils.generator_handler import GeneratorHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.utils.logger import logger


class TabularMetrics:
    """Class for computing Tabular metrics."""

    def __init__(self,
                 handler: MINVHandler,
                 generator_handler: GeneratorHandler,
                 configs: dict,
                 labels: torch.tensor = None,
                 z: torch.tensor = None,) -> None:
        """Initialize the TabularMetrics class."""
        self.handler = handler
        self.generator_handler = generator_handler
        self.generator = self.generator_handler.get_generator()
        self.target_model = self.handler.target_model

        self.labels = labels
        self.z = z
        logger.info("Configuring TabularMetrics")
        self._configure_metrics(configs)

        self.load_evaluation_model()

        self.test_dict = {
            "accuracy": self.compute_accuracy,
            "quality_metrics": self.quality_metrics,
            "plot_densities": self.get_numerical_density_plots,
            "plot_categorical_densities": self.get_categorical_density_plots,
            "gm_likelihood": self.gm_likelihood,
            "gower_distance": self.gower_dist,
            "plot_distibution": self.plot_distibution,
            "gower_report": self.gower_report_no_inv_loss
        }
        logger.info(configs)
        self.results = {}
        self.numerical_plots = {}
        self.categorical_plots = {}

        self.private_dataloader = self.handler.get_private_dataloader(self.batch_size)
        self.generated_samples = None
        # Compute desired metrics from configs
        # TODO: Change table_name)
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data=self.private_dataloader.dataset)
        # Get columns that are numerical in metadata
        self.numerical_columns = self.metadata.get_column_names(sdtype="numerical")
        self.categorical_columns = self.metadata.get_column_names(sdtype="categorical")
        self.best_rows = pd.DataFrame()
        self.metric_scheduler()

    def load_evaluation_model(self) -> None:
        """Load the evaluation model."""
        model_class = self.configs.eval_model.model_class
        if model_class is None:
            raise ValueError("model_class not found in configs.")

        module_path=self.configs.eval_model.module_path
        if module_path is None:
            raise ValueError("module_path not found in configs.")

        try:
            eval_module = import_module_from_file(module_path)
            self.eval_model_blueprint = get_class_from_module(eval_module, model_class)
            logger.info(f"Eval model blueprint created from {model_class} in {module_path}.")
        except Exception as e:
            raise ValueError(f"Failed to create the eval model blueprint from {model_class} in {module_path}") from e

        """Get the eval model metadata from the trained model metadata file."""
        model_path = self.configs.eval_model.eval_folder

        """Get the trained eval model."""
        try:
            self.evaluation_model = self.eval_model_blueprint.load_model(model_path)
            logger.info(f"Loaded eval model from {model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the trained eval model at {model_path}") from e

    def _configure_metrics(self, configs: dict) -> None:
        """Configure the metrics parameters.

        Args:
        ----
            configs (dict): Configuration parameters for the metrics.

        """
        self.configs = configs
        self.batch_size = configs.batch_size
        self.num_class_samples = configs.num_class_samples
        self.num_audited_classes = configs.num_audited_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def metric_scheduler(self) -> None:
        """Schedule the metrics to be computed."""
        tests = self.configs.metrics
        # If tests empty, return
        if not tests:
            logger.warning("No tests specified in the config.")
            return

        for test in tests:
            if test in self.test_dict:
                self.test_dict[test]()
            else:
                logger.warning(f"Test {test} not found in the test dictionary.")


    def compute_accuracy(self) -> None:
        """Compute accuracy for generated samples.

        We generate samples for each pair of label and z, and compute the accuracy of the evaluation model on these samples.
        """

        logger.info("Computing accuracy for generated samples.")
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)
        self.generator.eval()
        self.generator.to(self.device)

        # TODO: Perhaps define a config above to account for all potential optional parameters
        try:
            num_runs = self.configs.metrics["accuracy"].get("num_runs")
        except AttributeError:
            num_runs = 1
        logger.info(f"Number of runs for accuracy: {num_runs}")

        accuracies = []
        for i in range(num_runs):
            correct_predictions = []
            logger.info(f"Run {i+1}/{num_runs} for accuracy.")
            for label_i, z_i in zip(self.labels, self.z):
                # generate samples for each pair of label and z
                generated_samples, _, _ = self.generator_handler.sample_from_generator(batch_size=self.num_class_samples+1, # TODO: Move to configs asserts, num_class_samples ge 2
                                                                                label=label_i,
                                                                                z=z_i)
                synthetic_label = generated_samples["pseudo_label"].values
                synthetic_label = torch.tensor(synthetic_label, dtype=torch.int).to(self.device)
                generated_samples = generated_samples.drop(columns=["pseudo_label"])
                output = self.evaluation_model(generated_samples)
                prediction = torch.argmax(output, dim=1)
                correct_predictions.append(prediction == synthetic_label)

            correct_predictions = torch.cat(correct_predictions).float()
            accuracies.append(correct_predictions.mean())

        # Compute the mean and std of the accuracies
        self.accuracy = torch.mean(torch.tensor(accuracies))
        self.accuracy_std = torch.std(torch.tensor(accuracies))

        logger.info(f"Mean accuracy: {self.accuracy.item()}")
        logger.info(f"Standard deviation: {self.accuracy_std.item()}")
        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()


    def quality_metrics(self) -> None:
        """Compute quality metrics for the generated samples.

        We generate random samples. In this function, we do not pass the labels and z to the generator.
        """
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)
        self.generator.eval()
        self.generator.to(self.device)

        # Match the number of samples in the private dataloader
        desired_num_samples = len(self.private_dataloader.dataset)
        self.generated_samples = self.generator.sample(desired_num_samples)

        # We need to change column "pseudo_label" to identity in generated samples to match synthethic and real data
        self.generated_samples["identity"] = self.generated_samples["pseudo_label"]
        self.generated_samples = self.generated_samples.drop(columns=["pseudo_label"])

        logger.info("Computing quality metrics for generated samples.")
        report = QualityReport()
        report.generate(
            real_data=self.private_dataloader.dataset,
            synthetic_data=self.generated_samples,
            metadata=self.metadata.to_dict(),
            verbose=True
        )
        self.results["quality_report"] = report


    def get_numerical_density_plots(self) -> None:
        """Plot the densities of the numerical columns.

        We run this function after quality_metrics.
        """

        if self.numerical_columns is None:
            logger.warning("No numerical columns found in the metadata.")
            return
        for col in self.numerical_columns:
            if col == "identity":
                continue
            self.numerical_plots[f"{col}_density_plot"] = get_column_plot(
                real_data=self.private_dataloader.dataset,
                synthetic_data=self.generated_samples,
                column_name=col,
                metadata=self.metadata
            )

    def get_categorical_density_plots(self) -> None:
        """Plot the densities of the categorical columns.

        We run this function after quality_metrics.
        """
        # categorical_columns = self.categorical_columns
        # TODO: Make categorical columns to be all columns that are not numerical
        categorical_columns = ["identity", "race", "insurance", "gender"]
        for col in categorical_columns:
            self.categorical_plots[f"{col}_bar_plot"] = get_column_plot(
                real_data=self.private_dataloader.dataset,
                synthetic_data=self.generated_samples,
                column_name=col,
                metadata=self.metadata,
                plot_type="bar"
            )

    def gm_likelihood(self) -> None:
        """Compute the likelihood of the generated samples using the Gaussian Mixture Model (GMM).

        We run this function after quality_metrics.
        """
        real_data = self.private_dataloader.dataset[self.numerical_columns].drop(columns=["identity"])
        synthetic_data = self.generated_samples[self.numerical_columns].drop(columns=["identity"])
        self.results["gm_likelihood"] = GMLogLikelihood.compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )

    def gower_dist(self) -> None:
        """Compute the Gower distance and find best matches between synthethic and real data."""

        if self.generated_samples is None:
            # Match the number of samples in the private dataloader
            desired_num_samples = len(self.private_dataloader.dataset)
            self.generated_samples = self.generator.sample(desired_num_samples)

        table_synthethic = self.generated_samples
        table_real = self.private_dataloader.dataset

        # For each unique value in 'identity' column
        unique_values = table_real["identity"].unique()

        for value in unique_values:
            logger.info(f"Finding best row for identity: {value}")
            # Filter the table for the current value
            syn_subset = table_synthethic[table_synthethic["pseudo_label"] == value].copy()
            real_subset = table_real[table_real["identity"] == value].copy()
            offset = len(syn_subset)

            # Concat the two tables
            table = pd.concat([syn_subset, real_subset], axis=0)

            # Compute the Gower distance
            gower_distance = gower.gower_matrix(table)

            # Keep Real-synthethic distances
            gower_distance = gower_distance[:offset, offset:]

            min_index = np.unravel_index(np.argmin(gower_distance, axis=None), gower_distance.shape)

            # Get the corresponding synthetic and real rows
            synthetic_row = syn_subset.iloc[min_index[0]]
            real_row = real_subset.iloc[min_index[1]]

            synthetic_row = pd.DataFrame(synthetic_row).T
            real_row = pd.DataFrame(real_row).T

            # Cat synthetic and real rows to best_rows dataframe
            self.best_rows = pd.concat([self.best_rows, synthetic_row, real_row], axis=0)
            self.best_rows = self.best_rows.reset_index(drop=True)

    def save(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Placeholder function to save the metrics to disk.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        """
        pass

    def plot_distibution(self) -> None:
        """
        For each synthetic row, find the nearest REAL row by Gower distance
        *within the same class label only*, group distances by that REAL class,
        and save:
        - a histogram PNG per class
        - raw distances per class (CSV)
        - histogram bin counts per class (CSV)
        - a global matches table and summary (CSV)
        All files go to ./gower_plots.
        """
        import os, re, time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import gower

        # --- fixed output dir ---
        out_dir = os.path.join(os.getcwd(), "gower_plots")
        os.makedirs(out_dir, exist_ok=True)

        def _slugify(x):
            s = str(x)
            return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]

        # 1) Ensure synthetic data exists; ensure we have a label column for syn
        if self.generated_samples is None:
            desired_num = len(self.private_dataloader.dataset)
            self.generated_samples = self.generator.sample(desired_num)

        syn_df = self.generated_samples.copy()
        if "identity" not in syn_df.columns and "pseudo_label" in syn_df.columns:
            syn_df["identity"] = syn_df["pseudo_label"]  # align naming
        syn_label_col = "identity" if "identity" in syn_df.columns else "pseudo_label"

        real_df = self.private_dataloader.dataset.copy()
        real_label_col = "identity"

        real_df = real_df.reset_index(drop=True)
        syn_df  = syn_df.reset_index(drop=True)

        # 2) Select features for Gower (intersection minus labels/IDs)
        common = set(real_df.columns).intersection(set(syn_df.columns))
        drop_cols = {"identity", "pseudo_label", "hadm_id", "subject_id"}
        feature_cols = [c for c in common if c not in drop_cols]
        if not feature_cols:
            raise ValueError("No feature columns available for Gower distance.")

        # 3) Make dtypes Gower-friendly (avoid pandas Categorical/nullable dtypes)
        def _coerce(df):
            X = df[feature_cols].copy()
            for c in X.columns:
                dt = str(X[c].dtype)
                if dt == "boolean":  # pandas nullable boolean -> object
                    X[c] = X[c].astype(object)
                elif dt in {"Int64", "Int32", "Int16"}:  # pandas nullable ints -> float
                    X[c] = X[c].astype(float)
            for c in X.select_dtypes(include=["category"]).columns:
                X[c] = X[c].astype(object)
            return X

        # 4) Iterate by class: compare synthetic subset only to real rows of SAME class
        labels = np.intersect1d(
            syn_df[syn_label_col].unique(),
            real_df[real_label_col].unique()
        )

        total_syn = int(sum((syn_df[syn_label_col] == cls).sum() for cls in labels))
        pbar = tqdm(total=total_syn, desc="Gower NN (same-class)", unit="syn", dynamic_ncols=True)

        matches = []
        chunk_size = 500  # adjust based on RAM
        for cls in labels:
            syn_c_idx  = np.where(syn_df[syn_label_col].to_numpy() == cls)[0]
            real_c_idx = np.where(real_df[real_label_col].to_numpy() == cls)[0]
            if syn_c_idx.size == 0 or real_c_idx.size == 0:
                continue

            Xs_c = _coerce(syn_df.iloc[syn_c_idx])
            Xr_c = _coerce(real_df.iloc[real_c_idx])

            n_syn_c  = len(Xs_c)
            n_real_c = len(Xr_c)

            for start in range(0, n_syn_c, chunk_size):
                stop = min(start + chunk_size, n_syn_c)
                block = Xs_c.iloc[start:stop]
                bs = stop - start

                t0 = time.perf_counter()
                D = gower.gower_matrix(block, Xr_c)  # (bs x n_real_c)
                dt = time.perf_counter() - t0

                jmin = np.argmin(D, axis=1)
                dmin = D[np.arange(bs), jmin]

                real_idx_global = real_c_idx[jmin]
                syn_idx_global  = syn_c_idx[start:stop]

                matches.append(pd.DataFrame({
                    "syn_index": syn_idx_global,
                    "nearest_real_index": real_idx_global,
                    "distance": dmin,
                    "real_class": np.full(bs, cls),
                }))

                pairs_per_sec = (bs * n_real_c) / dt if dt > 0 else float("inf")
                pbar.set_postfix_str(f"class={cls} | {pairs_per_sec/1e6:.2f}M pairs/s, block {bs}")
                pbar.update(bs)

        pbar.close()
        matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(
            columns=["syn_index", "nearest_real_index", "distance", "real_class"]
        )
        self.results["gower_matches"] = matches

        # 5) Save histogram per class + save data behind each plot
        bins = 30
        hist_paths = {}
        raw_paths  = {}
        binned_paths = {}

        for cls, grp in matches.groupby("real_class"):
            vals = grp["distance"].values

            # (a) PNG plot
            fig = plt.figure()
            n, bin_edges, _ = plt.hist(vals, bins=bins)
            plt.xlabel("Gower distance to nearest REAL (same class)")
            plt.ylabel("Count")
            plt.title(f"Nearest-distance distribution (class = {cls})")
            plt.tight_layout()

            base = _slugify(cls)
            png_path = os.path.join(out_dir, f"gower_hist_{base}.png")
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            hist_paths[str(cls)] = png_path

            # (b) raw distances for the class
            raw_df = pd.DataFrame({"distance": vals})
            raw_path = os.path.join(out_dir, f"gower_distances_{base}.csv")
            raw_df.to_csv(raw_path, index=False)
            raw_paths[str(cls)] = raw_path

            # (c) histogram bin counts/edges for reproducibility
            # bin_edges has length bins+1; create left/right edges per bin
            binned_df = pd.DataFrame({
                "bin_left":  bin_edges[:-1],
                "bin_right": bin_edges[1:],
                "count":     n.astype(int),
            })
            binned_path = os.path.join(out_dir, f"gower_hist_data_{base}.csv")
            binned_df.to_csv(binned_path, index=False)
            binned_paths[str(cls)] = binned_path

        self.results["gower_hist_paths"]          = hist_paths
        self.results["gower_distance_raw_paths"]  = raw_paths
        self.results["gower_hist_data_paths"]     = binned_paths

        # 6) Save per-class stats + global matches
        summary = (matches.groupby("real_class")["distance"]
                        .agg(["count", "mean", "std", "median", "min", "max"])
                        .reset_index())
        summary_path = os.path.join(out_dir, "gower_distance_summary_same_class.csv")
        summary.to_csv(summary_path, index=False)
        self.results["gower_distance_summary"] = summary
        self.results["gower_distance_summary_path"] = summary_path

        matches_path = os.path.join(out_dir, "gower_matches_same_class.csv")
        matches.to_csv(matches_path, index=False)
        self.results["gower_matches_path"] = matches_path


    def gower_report_no_inv_loss(self) -> None:
        """
        One-shot Gower analysis (global NN):
        - For each synthetic row, find nearest REAL row by Gower distance over the FULL real set.
        - Per synthetic class: save histogram (PNG), raw distances (CSV), bin data (CSV).
        - Save a global matches CSV and per-class summary stats.
        - Save a cross-tab: for each SYN class, counts of the REAL classes picked as nearest.
        - Save same-class counts/ratios.
        - Save best (closest) syn–real pair per SYN class as self.best_rows (+ CSV).
        Outputs -> ./gower_plots ; paths & tables recorded in self.results.
        """
        import os, re, time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import gower

        # -------- output dir --------
        out_dir = os.path.join(os.getcwd(), "gower_plots")
        os.makedirs(out_dir, exist_ok=True)

        def _slugify(x):
            s = str(x)
            return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]

        # -------- ensure synthetic data exists & labels aligned --------
        if self.generated_samples is None:
            desired_num = len(self.private_dataloader.dataset)
            self.generated_samples = self.generator.sample(desired_num)

        syn_df = self.generated_samples.copy()
        if "identity" not in syn_df.columns and "pseudo_label" in syn_df.columns:
            syn_df["identity"] = syn_df["pseudo_label"]  # align naming
        syn_label_col = "identity" if "identity" in syn_df.columns else "pseudo_label"

        real_df = self.private_dataloader.dataset.copy()
        real_label_col = "identity"

        real_df = real_df.reset_index(drop=True)
        syn_df  = syn_df.reset_index(drop=True)

        # -------- features for Gower (intersection minus labels/IDs) --------
        common = set(real_df.columns).intersection(set(syn_df.columns))
        drop_cols = {"identity", "pseudo_label", "hadm_id", "subject_id"}
        feature_cols = [c for c in common if c not in drop_cols]
        if not feature_cols:
            raise ValueError("No feature columns available for Gower distance.")

        # -------- coerce dtypes so gower can handle them --------
        def _coerce(df):
            X = df[feature_cols].copy()
            # pandas nullable -> plain types
            for c in X.columns:
                dt = str(X[c].dtype)
                if dt == "boolean":
                    X[c] = X[c].astype(object)  # treat as categorical/object
                elif dt in {"Int64", "Int32", "Int16"}:
                    X[c] = X[c].astype(float)
            # pandas Categorical -> object
            for c in X.select_dtypes(include=["category"]).columns:
                X[c] = X[c].astype(object)
            return X

        Xr = _coerce(real_df)
        Xs = _coerce(syn_df)

        syn_classes = syn_df[syn_label_col].to_numpy()
        real_classes = real_df[real_label_col].to_numpy()

        # -------- global nearest neighbor (chunked over synthetic) --------
        matches = []  # rows: syn_index, nearest_real_index, distance, syn_class, real_class
        per_syn_class_best = {}  # syn_class -> dict(syn_index, real_index, distance)
        self.best_rows = pd.DataFrame()

        n_syn = len(Xs)
        n_real = len(Xr)
        chunk_size = 500  # adjust for RAM
        pbar = tqdm(total=n_syn, desc="Gower NN (global)", unit="syn", dynamic_ncols=True)

        for start in range(0, n_syn, chunk_size):
            stop = min(start + chunk_size, n_syn)
            block = Xs.iloc[start:stop]
            bs = stop - start

            t0 = time.perf_counter()
            D = gower.gower_matrix(block, Xr)  # (bs x n_real)
            dt = time.perf_counter() - t0

            jmin = np.argmin(D, axis=1)
            dmin = D[np.arange(bs), jmin]

            syn_idx_global = np.arange(start, stop)
            real_idx_global = jmin  # already aligned to Xr (global real_df index)
            syn_cls_block = syn_classes[syn_idx_global]
            real_cls_block = real_classes[real_idx_global]

            matches.append(pd.DataFrame({
                "syn_index": syn_idx_global,
                "nearest_real_index": real_idx_global,
                "distance": dmin,
                "syn_class": syn_cls_block,
                "real_class": real_cls_block,
            }))

            # update per synthetic class best pair (minimum distance) using this block
            for i in range(bs):
                cls = syn_cls_block[i]
                dist = float(dmin[i])
                if (cls not in per_syn_class_best) or (dist < per_syn_class_best[cls]["distance"]):
                    per_syn_class_best[cls] = dict(
                        syn_index=int(syn_idx_global[i]),
                        real_index=int(real_idx_global[i]),
                        distance=dist,
                    )

            pairs_per_sec = (bs * n_real) / dt if dt > 0 else float("inf")
            pbar.set_postfix_str(f"{pairs_per_sec/1e6:.2f}M pairs/s, block {bs}")
            pbar.update(bs)

        pbar.close()

        matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(
            columns=["syn_index", "nearest_real_index", "distance", "syn_class", "real_class"]
        )
        self.results["gower_matches"] = matches

        # -------- build self.best_rows (closest syn–real per synthetic class) --------
        best_rows_list = []
        for cls, rec in per_syn_class_best.items():
            syn_row  = syn_df.iloc[[rec["syn_index"]]].copy()
            real_row = real_df.iloc[[rec["real_index"]]].copy()
            syn_row["row_type"]  = "synthetic_best"
            real_row["row_type"] = "real_best"
            best_rows_list.extend([syn_row, real_row])
        self.best_rows = pd.concat(best_rows_list, ignore_index=True) if best_rows_list else pd.DataFrame()
        self.results["gower_per_syn_class_best"] = per_syn_class_best

        # -------- per synthetic class: save histogram/raw/bin data --------
        hist_paths, raw_paths, binned_paths = {}, {}, {}
        bins = 30
        for cls, grp in matches.groupby("syn_class"):
            vals = grp["distance"].values

            # (a) PNG histogram
            fig = plt.figure()
            n, bin_edges, _ = plt.hist(vals, bins=bins)
            plt.xlabel("Gower distance to nearest REAL (global)")
            plt.ylabel("Count")
            plt.title(f"Nearest-distance distribution (synthetic class = {cls})")
            plt.tight_layout()

            base = _slugify(cls)
            png_path = os.path.join(out_dir, f"gower_hist_synClass_{base}.png")
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            hist_paths[str(cls)] = png_path

            # (b) raw distances for this synthetic class
            raw_df = pd.DataFrame({"distance": vals})
            raw_path = os.path.join(out_dir, f"gower_distances_synClass_{base}.csv")
            raw_df.to_csv(raw_path, index=False)
            raw_paths[str(cls)] = raw_path

            # (c) histogram bin counts/edges
            binned_df = pd.DataFrame({
                "bin_left":  bin_edges[:-1],
                "bin_right": bin_edges[1:],
                "count":     n.astype(int),
            })
            binned_path = os.path.join(out_dir, f"gower_hist_data_synClass_{base}.csv")
            binned_df.to_csv(binned_path, index=False)
            binned_paths[str(cls)] = binned_path

        self.results["gower_hist_paths"]         = hist_paths
        self.results["gower_distance_raw_paths"] = raw_paths
        self.results["gower_hist_data_paths"]    = binned_paths

        # -------- summary stats by synthetic class --------
        summary = (matches.groupby("syn_class")["distance"]
                        .agg(["count", "mean", "std", "median", "min", "max"])
                        .reset_index())
        summary_path  = os.path.join(out_dir, "gower_distance_summary_by_syn_class.csv")
        summary.to_csv(summary_path, index=False)
        self.results["gower_distance_summary"]      = summary
        self.results["gower_distance_summary_path"] = summary_path

        # -------- cross-tab: nearest REAL class counts per SYN class --------
        nn_counts = (matches
                    .groupby(["syn_class", "real_class"])
                    .size()
                    .reset_index(name="count"))
        nn_pivot = nn_counts.pivot(index="syn_class", columns="real_class", values="count").fillna(0).astype(int)
        nn_counts_path = os.path.join(out_dir, "gower_nn_counts_synClass_vs_realClass.csv")
        nn_pivot.to_csv(nn_counts_path)
        self.results["gower_nn_counts"]      = nn_pivot
        self.results["gower_nn_counts_path"] = nn_counts_path

        # same-class counts/ratio per synthetic class
        same = nn_counts[nn_counts["syn_class"] == nn_counts["real_class"]].copy()
        same.rename(columns={"count": "same_class_count"}, inplace=True)
        tot = matches.groupby("syn_class").size().reset_index(name="total")
        same = tot.merge(same[["syn_class", "same_class_count"]], on="syn_class", how="left").fillna(0)
        same["same_class_count"] = same["same_class_count"].astype(int)
        same["same_class_ratio"] = same["same_class_count"] / same["total"]
        same_path = os.path.join(out_dir, "gower_nn_same_class_counts.csv")
        same.to_csv(same_path, index=False)
        self.results["gower_nn_same_class_counts"]      = same
        self.results["gower_nn_same_class_counts_path"] = same_path

        # -------- save global matches & best rows --------
        matches_path = os.path.join(out_dir, "gower_matches_global.csv")
        matches.to_csv(matches_path, index=False)
        self.results["gower_matches_path"] = matches_path

        if not self.best_rows.empty:
            best_rows_path = os.path.join(out_dir, "gower_best_rows_by_syn_class.csv")
            self.best_rows.to_csv(best_rows_path, index=False)
            self.results["gower_best_rows_path"] = best_rows_path
        else:
            self.results["gower_best_rows_path"] = None

    # def gower_report_with_inv_loss(self) -> None:
    #     """
    #     One-shot Gower analysis:
    #     - For each synthetic row, find nearest REAL row by Gower distance within the SAME class.
    #     - Build & save per-class histograms (PNG) + raw distances (CSV) + histogram bin data (CSV).
    #     - Find the single best (closest) syn–real pair per class and store both rows in self.best_rows.
    #     - Save a global matches table and per-class summary stats.
    #     Outputs go to ./gower_plots and paths are stored in self.results.
    #     """
    #     import os, re, time
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     from tqdm import tqdm
    #     import gower

    #     # --- fixed output dir ---
    #     out_dir = os.path.join(os.getcwd(), "gower_plots")
    #     os.makedirs(out_dir, exist_ok=True)

    #     def _slugify(x):
    #         s = str(x)
    #         return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]

    #     # 1) Ensure synthetic data exists; ensure we have a label column on synthetic
    #     if self.generated_samples is None:
    #         desired_num = len(self.private_dataloader.dataset)
    #         self.generated_samples = self.generator.sample(desired_num)

    #     syn_df = self.generated_samples.copy()
    #     if "identity" not in syn_df.columns and "pseudo_label" in syn_df.columns:
    #         syn_df["identity"] = syn_df["pseudo_label"]  # align naming
    #     syn_label_col = "identity" if "identity" in syn_df.columns else "pseudo_label"

    #     real_df = self.private_dataloader.dataset.copy()
    #     real_label_col = "identity"

    #     real_df = real_df.reset_index(drop=True)
    #     syn_df  = syn_df.reset_index(drop=True)

    #     # 2) Choose feature columns (intersection minus labels/IDs)
    #     common = set(real_df.columns).intersection(set(syn_df.columns))
    #     drop_cols = {"identity", "pseudo_label", "hadm_id", "subject_id"}
    #     feature_cols = [c for c in common if c not in drop_cols]
    #     if not feature_cols:
    #         raise ValueError("No feature columns available for Gower distance.")

    #     # 3) Make dtypes Gower-friendly (avoid pandas Categorical/nullable dtypes)
    #     def _coerce(df):
    #         X = df[feature_cols].copy()
    #         for c in X.columns:
    #             dt = str(X[c].dtype)
    #             if dt == "boolean":              # pandas nullable boolean -> object
    #                 X[c] = X[c].astype(object)
    #             elif dt in {"Int64", "Int32", "Int16"}:  # pandas nullable ints -> float
    #                 X[c] = X[c].astype(float)
    #         for c in X.select_dtypes(include=["category"]).columns:
    #             X[c] = X[c].astype(object)
    #         return X

    #     # 4) Iterate by class: nearest-only within SAME class, with progress bar
    #     labels = np.intersect1d(
    #         syn_df[syn_label_col].unique(),
    #         real_df[real_label_col].unique()
    #     )

    #     total_syn = int(sum((syn_df[syn_label_col] == cls).sum() for cls in labels))
    #     pbar = tqdm(total=total_syn, desc="Gower NN (same-class)", unit="syn", dynamic_ncols=True)

    #     matches = []  # per-synthetic nearest neighbor rows (indices, dist, class)
    #     best_rows_list = []  # to build self.best_rows (syn+real per class)
    #     per_class_best = {}  # keep best min distance and indices per class

    #     chunk_size = 500  # adjust based on RAM
    #     for cls in labels:
    #         syn_c_idx  = np.where(syn_df[syn_label_col].to_numpy() == cls)[0]
    #         real_c_idx = np.where(real_df[real_label_col].to_numpy() == cls)[0]
    #         if syn_c_idx.size == 0 or real_c_idx.size == 0:
    #             continue

    #         Xs_c = _coerce(syn_df.iloc[syn_c_idx])
    #         Xr_c = _coerce(real_df.iloc[real_c_idx])

    #         n_syn_c  = len(Xs_c)
    #         n_real_c = len(Xr_c)

    #         # Track best (closest) pair for this class (for best_rows like old gower_dist)
    #         best_dist_c = np.inf
    #         best_syn_glob = None
    #         best_real_glob = None

    #         for start in range(0, n_syn_c, chunk_size):
    #             stop = min(start + chunk_size, n_syn_c)
    #             block = Xs_c.iloc[start:stop]
    #             bs = stop - start

    #             t0 = time.perf_counter()
    #             D = gower.gower_matrix(block, Xr_c)  # (bs x n_real_c)
    #             dt = time.perf_counter() - t0

    #             jmin = np.argmin(D, axis=1)         # nearest real per synthetic (block)
    #             dmin = D[np.arange(bs), jmin]

    #             # Map back to global indices
    #             real_idx_global = real_c_idx[jmin]
    #             syn_idx_global  = syn_c_idx[start:stop]

    #             matches.append(pd.DataFrame({
    #                 "syn_index": syn_idx_global,
    #                 "nearest_real_index": real_idx_global,
    #                 "distance": dmin,
    #                 "real_class": np.full(bs, cls),
    #             }))

    #             # Update class-best pair using full D (not only per-row mins)
    #             flat_min = D.min()
    #             if flat_min < best_dist_c:
    #                 # find argmin coordinates in block, then map to global
    #                 bi, bj = np.unravel_index(D.argmin(), D.shape)
    #                 best_dist_c = float(flat_min)
    #                 best_syn_glob  = syn_c_idx[start + bi]
    #                 best_real_glob = real_c_idx[bj]

    #             # prog bar info
    #             pairs_per_sec = (bs * n_real_c) / dt if dt > 0 else float("inf")
    #             pbar.set_postfix_str(f"class={cls} | {pairs_per_sec/1e6:.2f}M pairs/s, block {bs}")
    #             pbar.update(bs)

    #         # stash best for this class
    #         if best_syn_glob is not None and best_real_glob is not None:
    #             per_class_best[int(cls)] = dict(
    #                 syn_index=int(best_syn_glob),
    #                 real_index=int(best_real_glob),
    #                 distance=float(best_dist_c),
    #             )
    #             # collect the actual rows (synthetic + real) as in old gower_dist
    #             syn_row  = syn_df.iloc[[best_syn_glob]].copy()
    #             real_row = real_df.iloc[[best_real_glob]].copy()
    #             best_rows_list.append(syn_row)
    #             best_rows_list.append(real_row)

    #     pbar.close()

    #     matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(
    #         columns=["syn_index", "nearest_real_index", "distance", "real_class"]
    #     )
    #     self.results["gower_matches"] = matches

    #     # 5) Build self.best_rows (synthetic + real rows, interleaved per class)
    #     self.best_rows = pd.concat(best_rows_list, ignore_index=True) if best_rows_list else pd.DataFrame()
    #     self.results["gower_per_class_best"] = per_class_best

    #     # 6) Save per-class histograms + raw distances + histogram bin data
    #     hist_paths, raw_paths, binned_paths = {}, {}, {}
    #     bins = 30
    #     for cls, grp in matches.groupby("real_class"):
    #         vals = grp["distance"].values

    #         # (a) PNG plot
    #         fig = plt.figure()
    #         n, bin_edges, _ = plt.hist(vals, bins=bins)
    #         plt.xlabel("Gower distance to nearest REAL (same class)")
    #         plt.ylabel("Count")
    #         plt.title(f"Nearest-distance distribution (class = {cls})")
    #         plt.tight_layout()

    #         base = _slugify(cls)
    #         png_path = os.path.join(out_dir, f"gower_hist_{base}.png")
    #         fig.savefig(png_path, dpi=200, bbox_inches="tight")
    #         plt.close(fig)
    #         hist_paths[str(cls)] = png_path

    #         # (b) raw distances for this class
    #         raw_df = pd.DataFrame({"distance": vals})
    #         raw_path = os.path.join(out_dir, f"gower_distances_{base}.csv")
    #         raw_df.to_csv(raw_path, index=False)
    #         raw_paths[str(cls)] = raw_path

    #         # (c) histogram bin counts/edges
    #         binned_df = pd.DataFrame({
    #             "bin_left":  bin_edges[:-1],
    #             "bin_right": bin_edges[1:],
    #             "count":     n.astype(int),
    #         })
    #         binned_path = os.path.join(out_dir, f"gower_hist_data_{base}.csv")
    #         binned_df.to_csv(binned_path, index=False)
    #         binned_paths[str(cls)] = binned_path

    #     self.results["gower_hist_paths"]         = hist_paths
    #     self.results["gower_distance_raw_paths"] = raw_paths
    #     self.results["gower_hist_data_paths"]    = binned_paths

    #     # 7) Save per-class stats + global matches
    #     summary = (matches.groupby("real_class")["distance"]
    #                     .agg(["count", "mean", "std", "median", "min", "max"])
    #                     .reset_index())
    #     summary_path  = os.path.join(out_dir, "gower_distance_summary_same_class.csv")
    #     matches_path  = os.path.join(out_dir, "gower_matches_same_class.csv")
    #     best_rows_path= os.path.join(out_dir, "gower_best_rows_same_class.csv")

    #     summary.to_csv(summary_path, index=False)
    #     matches.to_csv(matches_path, index=False)
    #     if not self.best_rows.empty:
    #         self.best_rows.to_csv(best_rows_path, index=False)
    #     else:
    #         best_rows_path = None

    #     self.results["gower_distance_summary"]      = summary
    #     self.results["gower_distance_summary_path"] = summary_path
    #     self.results["gower_matches_path"]          = matches_path
    #     self.results["gower_best_rows_path"]        = best_rows_path
