# LeakPro

![Tests](https://github.com/aidotse/LeakPro/actions/workflows/run_tests.yml/badge.svg)
![Last Commit](https://img.shields.io/github/last-commit/aidotse/LeakPro)
![License](https://img.shields.io/github/license/aidotse/LeakPro)
![Open Issues](https://img.shields.io/github/issues/aidotse/LeakPro)
![Open PRs](https://img.shields.io/github/issues-pr/aidotse/LeakPro)
![Downloads](https://img.shields.io/github/downloads/aidotse/LeakPro/total)
![Coverage](https://github.com/aidotse/LeakPro/blob/gh-pages/coverage.svg)

## About the project
The goal of LeakPro is to enable practitioners to seamlessly estimate the risk of leaking sensitive data when sharing machine learning models or synthetic datasets. 
LeakPro was created to bridge the gap between technical risk and legal risk, a challenge faced by many organizations today.
To achieve this, LeakPro provides tools to stress-test machine learning models and synthetic data by performing state-of-the-art privacy attacks. These include membership inference attacks, reconstruction attacks, and other advanced methods that assess the potential for sensitive information leakage.

The results are automatically collected, summarized, and presented in a comprehensive PDF report. This report is designed for easy sharing with stakeholders and to provide a solid foundation for risk assessment, compliance documentation, and decision-making around data sharing and model deployment.

## Privacy auditing

### Membership Inference Attacks (MIA)
![mia_flow](./resources/mia_flow.png) 
 

## Real world examples
# Real world examples

Please also check our industry use cases below. The use-cases cover four different data modalities, namely tabular, image, text, and graphs. 
Moreover ach use case 


<div align="center">

<table>
  <tr>
    <td align="center" width="400" height="200">
      <strong>Length-of-stay Prediction</strong><br>
      <img src="./resources/los.png" alt="LOS" style="width:150px; height:150px;">
      <br>
      <a href="length-of-stay.md">length-of-stay.md</a>
      <ul style="list-style:none; padding:0;">
        <li>Membership Inference: ✅</li>
        <li>Federated learning: ✅</li>
        <li>Synthetic data: ✅</li>
      </ul>
    </td>
    <td align="center" width="400" height="200">
      <strong>Text Masking</strong><br>
      <img src="./resources/NER.png" alt="NER" style="width:150px; height:150px;">
      <br>
      <a href="text-masking.md">text-masking.md</a>
      <ul style="list-style:none; padding:0;">
        <li>Membership Inference: ✅</li>
        <li>Federated learning: ✅</li>
        <li>Synthetic data: ✅</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td align="center" width="400" height="200">
      <strong>Camera Surveillance</strong><br>
      <img src="./resources/surveillance.png" alt="Surveillance" style="width:150px; height:150px;">
      <br>
      <a href="surveillance.md">surveillance.md</a>
      <ul style="list-style:none; padding:0;">
        <li>Membership Inference: ✅</li>
        <li>Federated learning: ✅</li>
        <li>Synthetic data: ❌</li>
      </ul>
    </td>
    <td align="center" width="400" height="200">
      <strong>Molecule Property Prediction</strong><br>
      <img src="./resources/graph.png" alt="Graph" style="width:150px; height:150px;">
      <br>
      <a href="molecule-property.md">molecule-property.md</a>
      <ul style="list-style:none; padding:0;">
        <li>Membership Inference: ✅</li>
        <li>Federated learning: ❌</li>
        <li>Synthetic data: ❌</li>
      </ul>
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
`pip install -e .`

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
