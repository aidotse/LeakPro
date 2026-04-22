import React, { useEffect, useState } from "react";
import Nav from "./components/Nav";
import Stepper from "./components/Stepper";
import Step1Upload from "./components/steps/Step1Upload";
import Step3Setup from "./components/steps/Step3Setup";
import Step4Models, { ModelEntry } from "./components/steps/Step4Models";
import Step5Attacks from "./components/steps/Step5Attacks";
import Step6Run from "./components/steps/Step6Run";
import Step7Results from "./components/steps/Step7Results";
import { api, ArchConfig, DataMeta } from "./api";

// Steps: 0=Dataset, 1=Setup, 2=Models, 3=Attacks, 4=Run, 5=Results
type Step = 0 | 1 | 2 | 3 | 4 | 5;

export default function App() {
  const [dark, setDark] = useState(false);
  const [step, setStep] = useState<Step>(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [dataMeta, setDataMeta] = useState<DataMeta | null>(null);
  const [archConfig, setArchConfig] = useState<ArchConfig | null>(null);
  const [models, setModels] = useState<ModelEntry[]>([]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    api.createJob().then((j) => setJobId(j.job_id));
  }, []);

  const restart = () => {
    api.createJob().then((j) => {
      setJobId(j.job_id);
      setStep(0);
      setDataMeta(null);
      setArchConfig(null);
      setModels([]);
    });
  };

  // Auto-confirm format from DataMeta and advance to Setup
  const handleDataDone = async (meta: DataMeta) => {
    setDataMeta(meta);
    if (jobId) {
      // Auto-confirm handler config — no user action needed
      await api.setHandlerConfig(jobId, {
        data_type: meta.data_type,
        shape: meta.shape,
        n_classes: meta.n_classes ?? 10,
        label_column: meta.label_column,
      });
    }
    setStep(1);
  };

  if (!jobId) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <span className="material-symbols-outlined text-4xl text-primary animate-spin">sync</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background-light dark:bg-background-dark text-slate-900 dark:text-slate-100 font-display">
      <Nav dark={dark} onToggleDark={() => setDark(!dark)} />

      <main className="flex-1 max-w-4xl mx-auto w-full px-6 py-12 flex flex-col gap-12">
        {step < 5 && <Stepper current={step} />}

        <div className="flex flex-col gap-8">
          {step === 0 && (
            <Step1Upload jobId={jobId} onDone={handleDataDone} />
          )}
          {step === 1 && dataMeta && (
            <Step3Setup
              jobId={jobId}
              handlerConfig={{ data_type: dataMeta.data_type, shape: dataMeta.shape, n_classes: dataMeta.n_classes ?? 10 }}
              onDone={(arch) => { setArchConfig(arch); setStep(2); }}
            />
          )}
          {step === 2 && (
            <Step4Models jobId={jobId} onDone={(m) => { setModels(m); setStep(3); }} />
          )}
          {step === 3 && (
            <Step5Attacks jobId={jobId} models={models} onDone={() => setStep(4)} />
          )}
          {step === 4 && (
            <Step6Run jobId={jobId} models={models} onDone={() => setStep(5)} />
          )}
          {step === 5 && (
            <Step7Results jobId={jobId} onRestart={restart} />
          )}
        </div>

        {step < 5 && (
          <div className="flex items-center justify-between pt-6 border-t border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm">
              <span className="material-symbols-outlined text-sm">lock</span>
              Your data is processed locally and never leaves your secure environment.
            </div>
            {step > 0 && (
              <button
                onClick={() => setStep((s) => (s - 1) as Step)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-sm font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                <span className="material-symbols-outlined text-base">arrow_back</span>
                Back
              </button>
            )}
          </div>
        )}
      </main>

      <div className="fixed bottom-0 left-0 w-full h-1 bg-gradient-to-r from-primary/10 via-primary/40 to-primary/10" />
    </div>
  );
}
