# Battery RUL Benchmark（2023）

The gradually increased amount of battery aging data has enabled rapid development of data-driven and machine learning algorithms in battery state assessment and lifetime prediction. Despite deep commitment and broad excitement, significant gaps still exist which hinders a thorough comparison and rapid iterative of the prediction algorithm. First, the formats across many public data sources are inconsistent; second, a majority of the algorithms are close source and lack of reproducibility; Last, the definition of evaluation metrices varies under different prediction scenarios, which pose challenges to compare different algorithms and develop novel one. 



If you do not have a Python environment locally, you can click on the `Colab` badge below and run the experimental example using the code we share.

<p align="center">
    <a href="https://colab.research.google.com/drive/1dqpDPpVbww95NOQ2JLZ2cKixyY_kPi_7?usp=sharing">
        <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://github.com/CXL-edu/Battery_RUL_Benchmark/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
    </a>
</p>






# Battery Datasets

Datasets available for battery RUL prediction tasks

| Data <br />Source | Chemistry<br/>of cathode | Nominal capacity and<br/>end of life(EOL) | Degration Characteristics |
| :--------- | :---------: | :---------: | :---------: |
| [NASA](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) | NCA | 2Ah/1.4Ah | Linear, Capacity recover |
| [CALCE](https://calce.umd.edu/battery-data) | LCO | 1.1Ah/0.88Ah | Linear, Have aging knee point |



# Prediction RUL



## Direct prediction of RUL in Nature dataset







## Iterative prediction of RUL using linear regression in NASA dataset



<div align=center><img src=".\file_to_readme\NASA\dynamic0.gif" alt="dynamic" width="399" height="228" /><img src=".\file_to_readme\NASA\dynamic22.gif" width="399" height="228" /></div>

<div align=center><img src=".\file_to_readme\NASA\dynamic32.gif" alt="dynamic" width="399" height="228" /><img src=".\file_to_readme\NASA\dynamic62.gif" width="399" height="228" /></div>



## Iterative prediction of RUL using Gaussian process regression method in NASA dataset



<div align=center><img src=".\file_to_readme\NASA\dynamic_GPR_0.gif" alt="dynamic" width="399" height="228" /><img src=".\file_to_readme\NASA\dynamic_GPR_52.gif" width="399" height="228" /></div>

<div align=center><img src=".\file_to_readme\NASA\GPR.png" width="630" height="300" /></div>

























