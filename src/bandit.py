import numpy as np


ARMS = {
    0: "Oferta A",
    1: "Oferta B",
    2: "Oferta C",
}


def get_arm_probability(row, base_probability, arm):
    """
    Calcula a probabilidade simulada de conversão
    para uma determinada estratégia.
    """

    probability = base_probability

    if arm == 0:
        effect = 0.00

    elif arm == 1:
        effect = 0.03

        if row["education"] == "tertiary":
            effect += 0.02

        if row["housing"] == "no":
            effect += 0.01

    elif arm == 2:
        effect = 0.02

        if row["poutcome"] == "success":
            effect += 0.03

        if row["previous"] > 0:
            effect += 0.01

    else:
        raise ValueError("Braço inválido")

    return float(np.clip(probability + effect, 0, 1))


class ThompsonSampling:

    def __init__(self, n_arms=3):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        samples = np.random.beta(
            self.alpha,
            self.beta
        )

        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1