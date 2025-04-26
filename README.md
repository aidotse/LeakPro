# LeakPro

![Tests](https://github.com/aidotse/LeakPro/actions/workflows/run_tests.yml/badge.svg)
![Last Commit](https://img.shields.io/github/last-commit/aidotse/LeakPro)
![License](https://img.shields.io/github/license/aidotse/LeakPro)
![Open Issues](https://img.shields.io/github/issues/aidotse/LeakPro)
![Open PRs](https://img.shields.io/github/issues-pr/aidotse/LeakPro)
![Downloads](https://img.shields.io/github/downloads/aidotse/LeakPro/total)
![Coverage](https://github.com/aidotse/LeakPro/blob/gh-pages/coverage.svg)

## About the project
<img align="left" width="250" height="250" src="https://github.com/aidotse/LeakPro/blob/main/resources/logo.jpg">


LeakPro was created to enable seamless risk assessment of leaking sensitive data when sharing machine learning models or synthetic datasets.  
To achieve this, it consolidates state-of-the-art privacy attacks into a unified and user-friendly tool, designed with a focus on realistic threat models and practical applicability.

When running LeakPro, results are automatically collected, summarized, and presented in a comprehensive PDF report. This report is designed for easy sharing with stakeholders and to provide a solid foundation for risk assessment, compliance documentation, and decision-making around data sharing and model deployment.

The recent [opinion from the EDPB](https://www.edpb.europa.eu/system/files/2024-12/edpb_opinion_202428_ai-models_en.pdf) has further underscored the necessity of a tool like LeakPro, emphasizing that to argue about model anonymity, a released model must have undergone stress-testing with “all means reasonably likely to be used” by an adversary.



## Philosophy behind LeakPro


LeakPro is built on the idea that privacy risks in machine learning can be framed as an [adversarial game](https://arxiv.org/abs/2212.10986) between a challenger and an attacker. In this framework, the attacker attempts to infer sensitive information from the challenger, while the challenger controls what information is exposed. By adjusting these controls, LeakPro allows users to explore different threat models, simulating various real-world attack scenarios.  

One common concern is that future attacks may surpass those currently known. To address this, LeakPro adopts a proactive approach, equipping adversaries with more side information than they would typically have in reality. This ensures that LeakPro does not just evaluate existing risks but also anticipates and tests against stronger, future threats, all while keeping assumptions realistic and relevant to practical scenarios. By integrating these principles, LeakPro serves as a flexible and robust tool for assessing privacy risks in machine learning models, helping researchers and practitioners stress-test their systems before real-world vulnerabilities emerge.  

LeakPro is designed to minimize user burden, requiring minimal manual input and featuring automated hyperparameter tuning for relevant attacks. The development is organized into four parallel legs with a shared architectural backbone:  
- **Membership Inference Attacks (MIA):**  
  This WP focuses on attacks that determine whether a specific data point was used in training. Adversaries in this setting have black-box access to the model, motivated by findings in the literature that black-box attacks can be as effective as white-box attacks.  

- **Model Inversion Attacks (MInvA):**  
  This recently initiated WP explores attacks that aim to reconstruct sensitive training data. In this case, the adversary is assumed to have white-box access to the model.  

- **Gradient Inversion Attacks (GIA):**  
  This WP targets federated learning, investigating the risk of an adversary reconstructing client data at the server by leveraging the global model and client updates.  

- **Synthetic Data Attacks:**  
  In this WP, adversaries only have access to a synthetic dataset generated from sensitive data. The goal is to infer information about the original dataset using only interactions with the synthetic data.  

Each leg follows core design principles: easy integration of new attacks, model agnosticism, and support for diverse data modalities. Currently, it supports tabular, image, text, and graph data, with time series integration underway.  


## Real world examples

Our [example portfolio](https://github.com/aidotse/LeakPro/tree/readme/examples) of real industry use cases cover four distinct data modalities: tabular, image, text, and graphs. The example portfolio is continuously improved and extended.

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


## Research Outputs  
LeakPro has contributed to the research community by enabling empirical studies on privacy risks in machine learning. Selected publications include:  

- [Krüger et al., *Publishing Neural Networks in Drug Discovery Might Compromise Training Data Privacy*, J. Cheminf, 2025](https://arxiv.org/abs/2410.16975)  
- [Brännvall et al., *Targeted Obfuscation for Machine Learning*, arXiv, 2024](https://arxiv.org/abs/2501.11525)
- [Reimer A., *Privacy Risks in Text Masking Models for Anonymization*, MSc Thesis, Chalmers University of Technology, 2025](https://odr.chalmers.se/server/api/core/bitstreams/3c2efd19-5440-43f7-a25f-051313f99c60/content)

## Credits
LeakPro draws inspiration from other works including
- [Murakonda S. K.,  Shokri .R, *ML Privacy Meter: Aiding Regulatory Compliance by Quantifying the Privacy Risks of Machine Learning*, arXiv, 2020] (https://github.com/privacytrustlab/ml_privacy_meter)
- [Giomi, M. et al., *A Unified Framework for Quantifying Privacy Risk in Synthetic Data, arXiv, 2022*](https://github.com/statice/anonymeter)

## Funding  
LeakPro is funded by Sweden's innovation agency, Vinnova, under grant 2023-03000. The project is a collaboration between AI Sweden, RISE, Scaleout AB, Syndata AB, AstraZeneca AB, Sahlgrenska University Hopsital, and Region Halland, with the goal of advancing privacy-preserving machine learning and responsible AI deployment.  
