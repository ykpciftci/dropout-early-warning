# Dropout Early Warning

Spotting the students with potential to drop out matters. These students sometimes can be helped as the trouble they face might be easy to resolve for management. Even when these students can not be helped, the data might be useful for management to plan resource allocation accordingly.

## Data

The dataset we are going to use is called **Predict Students' Dropout and Academic Success** and can be acquired from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

## Setup

You need Python 3.14

```bash
git clone https://github.com/ykpciftci/dropout-early-warning.git
cd dropout-early-warning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Download the dataset, unzip it, and place it at data/raw/data.csv