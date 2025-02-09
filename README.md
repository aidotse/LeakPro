# LeakPro

![Tests](https://github.com/aidotse/LeakPro/actions/workflows/run_tests.yml/badge.svg)
![Last Commit](https://img.shields.io/github/last-commit/aidotse/LeakPro)
![License](https://img.shields.io/github/license/aidotse/LeakPro)
![Open Issues](https://img.shields.io/github/issues/aidotse/LeakPro)
![Open PRs](https://img.shields.io/github/issues-pr/aidotse/LeakPro)
![Downloads](https://img.shields.io/github/downloads/aidotse/LeakPro/total)
![Coverage](https://github.com/aidotse/LeakPro/blob/gh-pages/coverage.svg)

## About the project
LeakPro was created to enable seamless risk assessment of leaking sensitive data when sharing machine learning models or synthetic datasets. 
To achieve this, it consolidates state-of-the-art privacy attacks into a unified and user-friendly tool, designed with a focus on realistic threat models and practical applicability.
LeakPro is also model agnostic (currently limited to uni-modal classification models) and currently supports four different data modalities (image, tabular, text, and graph).
There are four types of attacks supported, each attempting to answer a specific question:
- Membership Inference Attacks (MIA): "Is this datapoint part of the training data"?
- Model Inversion Attacks (MINVA): "What are the training data?" 
- Gradient Inversion Attacks (GIA): "In a federated learning scenario, given the global model and a local update, what was the training data used in the local update?"
- Synthetic Data (SynA): "Given a synthetic dataset, can sensitive information in the original data be inferred?"

When running LeakPro, results are automatically collected, summarized, and presented in a comprehensive PDF report. This report is designed for easy sharing with stakeholders and to provide a solid foundation for risk assessment, compliance documentation, and decision-making around data sharing and model deployment.
The recent [opinion from the EDPB](https://www.edpb.europa.eu/system/files/2024-12/edpb_opinion_202428_ai-models_en.pdf) has further underscored the necessity of a tool like LeakPro, emphasizing that to argue about model anonimity, a released model must have undergone stress-testing with “all means reasonably likely to be used” by an adversary.

### Philosophy behind LeakPro

Inspired by recent research [1], LeakPro conceptualizes privacy as a game between a **challenger** and an **attacker**. In this framework, the attacker attempts to infer sensitive information from the challenger. By controlling what is revealed to the attacker, different threat models can be explored.  

*A common concern is that future attacks may be stronger than those considered in LeakPro. To address this, we equip adversaries with more side information than what would typically be available in reality, while ensuring that the assumptions remain reasonable and contextually relevant to the scenario under consideration.*  

LeakPro is designed to **minimize user burden**, requiring minimal manual input and featuring **automated hyperparameter tuning** for relevant attacks. The development is organized into **four parallel work packages (WPs)** with a shared architectural backbone:  

- **Membership Inference Attacks (MIA):**  
  This WP focuses on attacks that determine whether a specific data point was used in training. Adversaries in this setting have **black-box access** to the model, motivated by findings in the literature that **black-box attacks can be as effective as white-box attacks**.  

- **Model Inversion Attacks (MINVA):**  
  This recently initiated WP explores attacks that aim to **reconstruct sensitive training data**. In this case, the adversary is assumed to have **white-box access** to the model.  

- **Gradient Inversion Attacks (GIA):**  
  This WP targets **federated learning**, investigating the risk of an adversary **reconstructing client data at the server** by leveraging the global model and client updates.  

- **Synthetic Data Attacks:**  
  In this WP, adversaries only have access to a **synthetic dataset** generated from sensitive data. The goal is to **infer information about the original dataset** using only interactions with the synthetic data.  

Each WP follows core design principles: **easy integration of new attacks, model agnosticism, and support for diverse data modalities**. Currently, it supports **tabular, image, text, and graph data**, with **time series integration underway**.  

Phase one focuses on **classification models** (ranging from **decision trees to deep neural networks**), with an **expansion to generative models** planned for **phase two in 2026**.  

Although LeakPro primarily focuses on **building a compliance tool**, its development has also advanced research, leading to **two publications in its first year** [2,3].  

---

**References**  
[1] Salem et al., 2023. *SOK: Privacy Attacks on Machine Learning Models.*  
[2] Krüger et al., 2024. *Publishing Neural Networks for Drug Discovery: Privacy Considerations.*  
[3] Brännvall et al., 2025. *Technical Report on the Forgotten by Design Project.*  



## Real world examples

Our industry use cases cover four distinct data modalities: tabular, image, text, and graphs. Each use case supports various types of privacy attacks, providing a comprehensive evaluation framework.

<div align="center">

<table>
  <tr>
    <td align="center" width="400" height="200">
      <strong>Length-of-stay Prediction</strong><br>
      <img src="./resources/los.png" alt="LOS" style="width:150px; height:150px;">
      <br>
      <a href="length-of-stay.md">length-of-stay.md</a>
      <div style="text-align: left;">
        MIA: ✅<br>
        MInvA: ❌<br>
        GIA: ✅</br>
        SynA: ✅
      </div>
    </td>
    <td align="center" width="400" height="200">
      <strong>Text Masking</strong><br>
      <img src="./resources/NER.png" alt="NER" style="width:150px; height:150px;">
      <br>
      <a href="text-masking.md">text-masking.md</a>
      <div style="text-align: left;">
        MIA: ✅<br>
        MInvA: ❌<br>
        GIA: ✅</br>
        SynA: ✅
      </div>
    </td>
  </tr>
  <tr>
    <td align="center" width="400" height="200">
      <strong>Camera Surveillance</strong><br>
      <img src="./resources/surveillance.png" alt="Surveillance" style="width:150px; height:150px;">
      <br>
      <a href="surveillance.md">surveillance.md</a>
      <div style="text-align: left;">
        MIA: ✅<br>
        MInvA: ❌<br>
        GIA: ✅</br>
        SynA: ❌
      </div>
    </td>
    <td align="center" width="400" height="200">
      <strong>Molecule Property Prediction</strong><br>
      <img src="./resources/graph.png" alt="Graph" style="width:150px; height:150px;">
      <br>
      <a href="molecule-property.md">molecule-property.md</a>
      <div style="text-align: left;">
        MIA: ✅<br>
        MInvA: ❌<br>
        GIA: ❌</br>
        SynA: ❌
      </div>
    </td>
  </tr>
</table>

</div>




## To install
0. **Clone repository**
`git clone https://github.com/aidotse/LeakPro.git`
1. **Navigate to the project repo**
`cd Leakpro`
2. **Install with pip**
`pip install -e .[dev]`

## To Contribute
0. **Ensure local repo is up-to-date:**
`git fetch origin`
2. **Create feature branch**
 `git checkout -b my-feature-branch`
3. **Make changes and commit:**
`git add . ` 
`git commit -m "Added new feature" `
4. **Ensure the local main is up to date:**
`git checkout main`
`git pull origin main`
5. **Merge main onto feature branch**
`git checkout my-feature-branch`
`git merge main`
6. **Resolve conflicts, add and commit.**
7. **Push your update to the remore repository**
`git push origin my-feature-branch`
8. **Open pull request**
