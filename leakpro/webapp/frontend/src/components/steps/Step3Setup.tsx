import React, { useState } from "react";
import ReactDOM from "react-dom";
import { api, ArchConfig, HandlerConfig } from "../../api";
import ServerOrUpload from "../ServerOrUpload";

// ---------------------------------------------------------------------------
// Code templates (inlined for download / preview)
// ---------------------------------------------------------------------------

const HANDLER_TEMPLATE = `"""
Training handler for LeakPro — defines the training loop for your model.

Data loading and normalisation belong in dataset_handler.py (uploaded in Step 1).
This file only needs to implement train() and eval().
"""

import torch
from torch import cuda, device, optim, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput


class MyInputHandler(AbstractInputHandler):
    """Implement train() and eval() for your model architecture."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        val_split = 0.1
        patience = 10
        dataset = dataloader.dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=dataloader.batch_size,
                                  shuffle=True, num_workers=dataloader.num_workers)
        val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size,
                                shuffle=False, num_workers=dataloader.num_workers)

        if epochs is None:
            raise ValueError("epochs not found in configs")

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history, loss_history = [], []
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pred = outputs.argmax(dim=1)
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)
                pbar.set_postfix(loss=f"{train_loss/total_samples:.4f}",
                                  acc=f"{train_acc/total_samples:.4f}")
            scheduler.step()

            accuracy_history.append(train_acc / total_samples)
            loss_history.append(train_loss / total_samples)

            # Validation + early stopping
            model.eval()
            val_loss, val_samples = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * labels.size(0)
                    val_samples += labels.size(0)
            avg_val_loss = val_loss / val_samples

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    model.load_state_dict(best_model_state)
                    break

        model.to("cpu")
        results = EvalOutput(
            accuracy=accuracy_history[-1],
            loss=loss_history[-1],
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion) -> EvalOutput:
        """Model evaluation procedure."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc, total_samples = 0, 0, 0
        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                pred = output.argmax(dim=1)
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
        return EvalOutput(accuracy=float(acc) / total_samples, loss=loss / total_samples)
`;

const ARCH_TEMPLATE = `"""Model architecture for LeakPro — define your nn.Module subclass here."""

import torch.nn as nn
import torch.nn.functional as F


# Define any helper classes above if needed (e.g. BasicBlock, NetworkBlock)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        # ... instantiate your layer blocks here (e.g. BasicBlock, NetworkBlock)
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = None  # replace with your block: NetworkBlock(n, nChannels[0], nChannels[1], ...)
        self.block2 = None  # replace with your block: NetworkBlock(n, nChannels[1], nChannels[2], ...)
        self.block3 = None  # replace with your block: NetworkBlock(n, nChannels[2], nChannels[3], ...)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
`;

// ---------------------------------------------------------------------------
// Preset definitions
// ---------------------------------------------------------------------------

interface PresetDetail {
  architecture: string;
  optimizer: string;
  learning_rate: string;
  scheduler: string;
}

const PRESETS: Array<{
  id: string; label: string; icon: string; desc: string;
  types: string[]; details: PresetDetail;
}> = [
  {
    id: "cifar_image",
    label: "Image (CIFAR-style)",
    icon: "image",
    desc: "WideResNet architecture with standard image augmentation. Works for 3-channel images (32×32 or larger).",
    types: ["image"],
    details: {
      architecture: "WideResNet (depth=28, widen_factor=2) — ~36M params, 3 residual blocks",
      optimizer: "SGD (momentum=0.9, weight_decay=5e-4)",
      learning_rate: "0.1",
      scheduler: "CosineAnnealingLR",
    },
  },
  {
    id: "tabular_mlp",
    label: "Tabular (MLP)",
    icon: "table_rows",
    desc: "Multi-layer perceptron for tabular / CSV data with numeric features.",
    types: ["tabular"],
    details: {
      architecture: "MLP — 3 hidden layers [256 → 128 → 64] + ReLU + Dropout(0.3)",
      optimizer: "Adam",
      learning_rate: "1e-3",
      scheduler: "None",
    },
  },
  {
    id: "time_series",
    label: "Time Series (GRU)",
    icon: "show_chart",
    desc: "GRU-based sequence model for time-series data.",
    types: ["time_series"],
    details: {
      architecture: "GRU (hidden=128, layers=2) + Linear head",
      optimizer: "Adam",
      learning_rate: "1e-3",
      scheduler: "None",
    },
  },
];

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface Props {
  jobId: string;
  handlerConfig: HandlerConfig;
  onDone: (arch: ArchConfig) => void;
}

export default function Step3Setup({ jobId, handlerConfig, onDone }: Props) {
  const [mode, setMode] = useState<"preset" | "upload">("preset");
  const [selectedPreset, setSelectedPreset] = useState<string | null>(
    PRESETS.find((p) => p.types.includes(handlerConfig.data_type))?.id ?? null
  );
  const [archFile, setArchFile] = useState<File | null>(null);
  const [handlerFile, setHandlerFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedPreset, setExpandedPreset] = useState<string | null>(null);
  const [modal, setModal] = useState<"arch" | "handler" | null>(null);

  const canProceed = mode === "preset" ? !!selectedPreset : !!(archFile && handlerFile);

  const proceed = async () => {
    setLoading(true);
    setError(null);
    try {
      if (mode === "upload" && archFile && handlerFile) {
        await api.uploadArch(jobId, archFile);
        await api.uploadHandler(jobId, handlerFile);
      }
      const config: ArchConfig = {
        preset: mode === "preset" ? selectedPreset! : undefined,
        arch_filename: archFile?.name,
        handler_filename: handlerFile?.name,
      };
      await api.setArchConfig(jobId, config);
      onDone(config);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="space-y-2">
        <h2 className="text-4xl font-black tracking-tight">Architecture & Training</h2>
        <p className="text-slate-600 dark:text-slate-400 text-lg max-w-2xl">
          Choose a built-in preset that matches your data type, or upload your own architecture
          and training loop.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2">
        {(["preset", "upload"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-5 py-2 rounded-lg font-bold text-sm transition-colors
              ${mode === m
                ? "bg-primary text-white shadow-lg shadow-primary/20"
                : "border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
          >
            {m === "preset" ? "Use built-in preset" : "Upload my own"}
          </button>
        ))}
      </div>

      {mode === "preset" ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {PRESETS.map((p) => {
            const enabled = p.types.includes(handlerConfig.data_type);
            const selected = selectedPreset === p.id;
            const expanded = expandedPreset === p.id;
            return (
              <div
                key={p.id}
                onClick={enabled ? () => setSelectedPreset(p.id) : undefined}
                className={`rounded-xl border text-left transition-all flex flex-col
                  ${enabled ? "cursor-pointer" : "opacity-40 cursor-not-allowed"}
                  ${selected
                    ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
                    : "border-slate-200 dark:border-slate-800"}
                  ${enabled && !selected ? "hover:border-primary/40" : ""}
                `}
              >
                <div className="p-5 flex flex-col gap-3 flex-1">
                  <div className="flex items-start gap-3">
                    <div className="size-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary shrink-0">
                      <span className="material-symbols-outlined">{p.icon}</span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-bold text-sm">{p.label}</p>
                      {enabled ? (
                        <span className="text-xs text-green-600 dark:text-green-400 font-semibold">
                          ✓ Recommended for your data
                        </span>
                      ) : (
                        <span className="text-xs text-slate-400 font-medium">
                          Requires {p.types[0]} data
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">{p.desc}</p>
                </div>

                {/* Details toggle */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedPreset(expanded ? null : p.id);
                  }}
                  disabled={!enabled}
                  className="w-full flex items-center justify-between px-5 py-2.5 border-t border-slate-100 dark:border-slate-800 text-xs font-semibold text-slate-500 hover:text-primary transition-colors disabled:pointer-events-none"
                >
                  Details
                  <span className="material-symbols-outlined text-sm">
                    {expanded ? "expand_less" : "expand_more"}
                  </span>
                </button>

                {/* Expanded details */}
                {expanded && (
                  <div className="px-5 pb-5 flex flex-col gap-2 border-t border-slate-100 dark:border-slate-800 pt-3">
                    <DetailRow label="Architecture" value={p.details.architecture} />
                    <DetailRow label="Optimizer" value={p.details.optimizer} />
                    <DetailRow label="Learning rate" value={p.details.learning_rate} />
                    <DetailRow label="Scheduler" value={p.details.scheduler} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Model Architecture (.py)
              </span>
              <button
                onClick={() => setModal("arch")}
                className="flex items-center gap-1 text-xs text-primary font-semibold hover:underline"
              >
                <span className="material-symbols-outlined text-sm">code</span>
                View example
              </button>
            </div>
            <ServerOrUpload
              label="Model Architecture (.py)"
              hint="Python file defining your nn.Module subclass"
              icon="code"
              accept=".py"
              onFile={async (f) => { await api.uploadArch(jobId, f); setArchFile(f); }}
              onPath={async (p) => { await api.setArchPath(jobId, p); setArchFile(new File([], p.split("/").pop() ?? "arch.py")); }}
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Training Loop / Handler (.py)
              </span>
              <button
                onClick={() => setModal("handler")}
                className="flex items-center gap-1 text-xs text-primary font-semibold hover:underline"
              >
                <span className="material-symbols-outlined text-sm">code</span>
                View example
              </button>
            </div>
            <ServerOrUpload
              label="Training Loop / Handler (.py)"
              hint="Python file with your training handler class"
              icon="settings"
              accept=".py"
              onFile={async (f) => { await api.uploadHandler(jobId, f); setHandlerFile(f); }}
              onPath={async (p) => { await api.setHandlerPath(jobId, p); setHandlerFile(new File([], p.split("/").pop() ?? "handler.py")); }}
            />
          </div>
        </div>
      )}

      {error && <p className="text-sm text-red-500">{error}</p>}

      <div className="flex justify-end">
        <button
          onClick={proceed}
          disabled={!canProceed || loading}
          className="px-8 py-2.5 rounded-lg bg-primary text-white font-bold hover:bg-primary/90 transition-colors flex items-center gap-2 shadow-lg shadow-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Saving…" : "Continue"}
          <span className="material-symbols-outlined text-base">arrow_forward</span>
        </button>
      </div>

      {/* Code preview modals */}
      {modal === "arch" && (
        <CodeModal
          title="Example: Model Architecture"
          caption="Your architecture file must define a PyTorch nn.Module subclass. The class name will be used to instantiate the model."
          code={ARCH_TEMPLATE}
          filename="arch_example.py"
          onClose={() => setModal(null)}
        />
      )}
      {modal === "handler" && (
        <CodeModal
          title="Example: Training Handler"
          caption="Your handler must implement a class with a train(model, dataset, **kwargs) method that returns a TrainingOutput."
          code={HANDLER_TEMPLATE}
          filename="handler_example.py"
          onClose={() => setModal(null)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-xs font-mono font-semibold text-slate-700 dark:text-slate-300">{value}</span>
    </div>
  );
}

function downloadFile(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function CodeModal({ title, caption, code, filename, onClose }: {
  title: string; caption: string; code: string; filename: string; onClose: () => void;
}) {
  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl w-full max-w-3xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800">
          <h3 className="font-bold text-lg">{title}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        {/* Caption */}
        <p className="px-6 pt-4 text-sm text-slate-500 dark:text-slate-400">{caption}</p>

        {/* Code */}
        <pre className="overflow-auto px-6 py-4 font-mono text-xs text-slate-300 bg-slate-950 mx-6 my-4 rounded-xl max-h-[55vh] leading-relaxed">
          {code}
        </pre>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-slate-200 dark:border-slate-800">
          <button
            onClick={() => downloadFile(code, filename)}
            className="flex items-center gap-2 px-5 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <span className="material-symbols-outlined text-base">download</span>
            Download template
          </button>
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-lg bg-primary text-white text-sm font-bold hover:bg-primary/90 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
