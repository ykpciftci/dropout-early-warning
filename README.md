# Dropout Early Warning
[![Tests](https://github.com/ykpciftci/dropout-early-warning/actions/workflows/tests.yml/badge.svg)](https://github.com/ykpciftci/dropout-early-warning/actions/workflows/tests.yml)

* Flags students at risk of dropping out. It runs for students at enrollment and at the end of semester 1.
* On the test set, semester 1 model catches 84% of dropouts and almost halves missed dropouts compared to predicting at enrollment. (85 -> 46)
* The app is built with scikit-learn and served with FastAPI. It is packaged with Docker and tested with CI.

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

## Approach

- We have grouped data according to the time the data become available within the first year of school. The groups were the features known at enrollment, features known at the end of 1st semester and features known after 2nd semester. 2 models were built, one for enrollment features and another for features known after semester 1.
- We were also suspicious about the timing of financial columns, in other words, we were not sure when the data become available to the management, this is why we excluded them from the training data. However we did experiment with them.
- Enrolled students were dropped, Only two classes left for target(Dropouts = 1, Graduates = 0).
- As a classification method, we used Logistic regression with balanced class weights.
- For model decisions we used 10x5 repeated stratified cross-validation.
- 20% test set was held out and only used once for evaluation at the end.


## Results

|Model        |CV Recall  |Test Recall  |CV F1   |Test F1 |Test Precision  | Missed Dropouts(of 284) |
|-------------|-----------|-------------|--------|--------|----------------|-------------------------|
|Enrollment   |0.682      |0.70         |0.652   |0.65    |0.60            |85                       |
|Semester 1   |0.817      |0.84         |0.816   |0.80    |0.77            |46                       |

- The test scores match our CV results which shows the models show no sign of overfitting.
- Waiting one semester almost halves the missed dropouts.
- Precision also rises from 0.60 to 0.77, meaning number of false flags drops sharply. This makes shortlists shorter and more often right.
- Missed dropouts drop from 85 to 46 and false alarms from about 130 to about 71, at the cost of half a year less time to act.

## Findings

- If the school can confirm that payment and scholarship status are known by the end of semester 1, these features could be added for a modest gain (+0.03 recall, about the size of the CV spread). if we can not be certain about data acquisition timing, such modest gain is not worth to trade off with leakage possibility.
- We observed that Semester 1 Model already handles the students with zero evaluations. 78% of these students drop out; the model catches 190 of 196 but also flags 49 of 55 graduates. That’s acceptable for such a high-risk group. We do not need to make further adjustments on the data.

## Usage

### Train
```bash
python -m src.train
```

### Evaluate
```bash
python -m src.evaluate
```

### Test
```bash
pytest
```
### Run the API
Make sure you have trained the models first (see Train above). models/ folder is not in the repo (gitignored).

- To start server:
```bash
uvicorn src.api:app --reload
```
- API endpoints can be accessed at http://127.0.0.1:8000/docs
- Endpoints:
    -  GET /health
    - POST /predict/enrollment: enrollment features only
    - POST /predict/semester-1: enrollment plus 1st-semester features

- Schema for sample JSON request body can be found at API endpoints docs page. Use "Try it out" button.
- JSON response will look like:
```
{"dropout": true, "dropout_risk_score":0.825}
```
-  dropout_risk_score is a risk score, not a calibrated probability, because of class_weight="balanced". The dropout flag uses 0.5 cutoff which we chose.

### Run with Docker
- Train the models first (see Train above). They aren't stored in repository, and image copies them in.

```bash
docker build -t dropout-app .
docker run -p 8000:8000 dropout-app
```
- API endpoints can be accessed at http://localhost:8000/docs
- Docker image includes trained models.
- requirements-api.txt contains the runtime dependencies used by the image, whereas requirements.txt is for the full development setup.

## Limitations

- The data was obtained from a Portuguese institution. More, less or different data will be available for any other country's students and its institutions.
- We did not account for students who are still enrolled, because their outcome is unknown; some of them may still drop out.
