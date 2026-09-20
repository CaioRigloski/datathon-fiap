from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os

from src.bandit import (
    ARMS,
    ThompsonSampling,
    get_arm_probability,
)

app = FastAPI(
    title="Adaptive Offer Recommendation API",
    description="API para recomendação adaptativa de ofertas",
    version="1.0.0",
)

model = joblib.load("src/propensity_model.pkl")

BANDIT_STATE_PATH = "src/bandit_state.json"


def load_bandit():
    bandit = ThompsonSampling(n_arms=len(ARMS))

    if os.path.exists(BANDIT_STATE_PATH):
        with open(BANDIT_STATE_PATH, "r", encoding="utf-8") as file:
            state = json.load(file)

        bandit.alpha = np.array(state["alpha"], dtype=float)
        bandit.beta = np.array(state["beta"], dtype=float)

    return bandit


def save_bandit(bandit):
    state = {
        "alpha": bandit.alpha.tolist(),
        "beta": bandit.beta.tolist(),
    }

    with open(BANDIT_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


bandit = load_bandit()


features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


class CustomerRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


class FeedbackRequest(BaseModel):
    arm: int
    reward: int


@app.get("/")
def root():
    return {
        "message": "Adaptive Offer Recommendation API",
        "status": "running",
    }


@app.post("/recommend")
def recommend(customer: CustomerRequest):
    customer_data = {
        "age": customer.age,
        "job": customer.job,
        "marital": customer.marital,
        "education": customer.education,
        "default": customer.default,
        "housing": customer.housing,
        "loan": customer.loan,
        "contact": customer.contact,
        "month": customer.month,
        "day_of_week": customer.day_of_week,
        "campaign": customer.campaign,
        "pdays": customer.pdays,
        "previous": customer.previous,
        "poutcome": customer.poutcome,
        "emp.var.rate": customer.emp_var_rate,
        "cons.price.idx": customer.cons_price_idx,
        "cons.conf.idx": customer.cons_conf_idx,
        "euribor3m": customer.euribor3m,
        "nr.employed": customer.nr_employed,
    }

    df = pd.DataFrame([customer_data])

    # Probabilidade base gerada pelo modelo de propensity
    base_probability = model.predict_proba(
        df[features]
    )[0, 1]

    # Probabilidade estimada para cada oferta
    arm_probabilities = {
        ARMS[arm]: get_arm_probability(
            df.iloc[0],
            base_probability,
            arm
        )
        for arm in ARMS
    }

    # Melhor oferta segundo a estimativa contextual
    best_estimated_offer = max(
        arm_probabilities,
        key=arm_probabilities.get
    )

    # Escolha feita pelo Thompson Sampling
    selected_arm = bandit.select_arm()
    thompson_sampling_offer = ARMS[selected_arm]

    return {
        "best_estimated_offer": best_estimated_offer,
        "thompson_sampling_offer": thompson_sampling_offer,
        "selected_arm": selected_arm,
        "base_probability": round(
            float(base_probability),
            4
        ),
        "offer_probabilities": {
            offer: round(float(probability), 4)
            for offer, probability in arm_probabilities.items()
        },
        "bandit_state": {
            "alpha": bandit.alpha.tolist(),
            "beta": bandit.beta.tolist(),
        },
    }


@app.post("/feedback")
def feedback(data: FeedbackRequest):
    if data.arm not in ARMS:
        raise HTTPException(
            status_code=400,
            detail="Braço inválido. Use 0, 1 ou 2."
        )

    if data.reward not in [0, 1]:
        raise HTTPException(
            status_code=400,
            detail="Reward deve ser 0 ou 1."
        )

    # Atualiza o Thompson Sampling
    bandit.update(
        data.arm,
        data.reward
    )

    # Persiste o aprendizado
    save_bandit(bandit)

    return {
        "message": "Feedback registrado com sucesso.",
        "arm": data.arm,
        "offer": ARMS[data.arm],
        "reward": data.reward,
        "bandit_state": {
            "alpha": bandit.alpha.tolist(),
            "beta": bandit.beta.tolist(),
        },
    }