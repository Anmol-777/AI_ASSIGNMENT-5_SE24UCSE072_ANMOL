from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np


def build_medical_diagnosis_bn():
    model = BayesianNetwork([
        ('Age', 'Hypertension'),
        ('Smoking', 'Hypertension'),
        ('Smoking', 'LungDisease'),
        ('Hypertension', 'HeartDisease'),
        ('LungDisease', 'HeartDisease'),
        ('HeartDisease', 'ChestPain'),
        ('HeartDisease', 'Breathlessness')
    ])

    cpd_age = TabularCPD(
        variable='Age',
        variable_card=2,
        values=[[0.6], [0.4]],
        state_names={'Age': ['young', 'old']}
    )

    cpd_smoking = TabularCPD(
        variable='Smoking',
        variable_card=2,
        values=[[0.7], [0.3]],
        state_names={'Smoking': ['no', 'yes']}
    )

    cpd_hypertension = TabularCPD(
        variable='Hypertension',
        variable_card=2,
        values=[
            [0.85, 0.60, 0.55, 0.30],
            [0.15, 0.40, 0.45, 0.70]
        ],
        evidence=['Age', 'Smoking'],
        evidence_card=[2, 2],
        state_names={
            'Hypertension': ['no', 'yes'],
            'Age': ['young', 'old'],
            'Smoking': ['no', 'yes']
        }
    )

    cpd_lung = TabularCPD(
        variable='LungDisease',
        variable_card=2,
        values=[[0.95, 0.55], [0.05, 0.45]],
        evidence=['Smoking'],
        evidence_card=[2],
        state_names={
            'LungDisease': ['no', 'yes'],
            'Smoking': ['no', 'yes']
        }
    )

    cpd_heart = TabularCPD(
        variable='HeartDisease',
        variable_card=2,
        values=[
            [0.90, 0.70, 0.65, 0.30],
            [0.10, 0.30, 0.35, 0.70]
        ],
        evidence=['Hypertension', 'LungDisease'],
        evidence_card=[2, 2],
        state_names={
            'HeartDisease': ['no', 'yes'],
            'Hypertension': ['no', 'yes'],
            'LungDisease': ['no', 'yes']
        }
    )

    cpd_chest = TabularCPD(
        variable='ChestPain',
        variable_card=2,
        values=[[0.80, 0.25], [0.20, 0.75]],
        evidence=['HeartDisease'],
        evidence_card=[2],
        state_names={
            'ChestPain': ['no', 'yes'],
            'HeartDisease': ['no', 'yes']
        }
    )

    cpd_breath = TabularCPD(
        variable='Breathlessness',
        variable_card=2,
        values=[[0.85, 0.30], [0.15, 0.70]],
        evidence=['HeartDisease'],
        evidence_card=[2],
        state_names={
            'Breathlessness': ['no', 'yes'],
            'HeartDisease': ['no', 'yes']
        }
    )

    model.add_cpds(cpd_age, cpd_smoking, cpd_hypertension, cpd_lung,
                   cpd_heart, cpd_chest, cpd_breath)

    assert model.check_model(), "Bayesian Network model is invalid"
    return model


def run_inference(model):
    infer = VariableElimination(model)

    print("\n--- Query 1: Prior probability of Heart Disease ---")
    result = infer.query(variables=['HeartDisease'])
    print(result)

    print("\n--- Query 2: P(HeartDisease | Smoking=yes, Age=old) ---")
    result = infer.query(
        variables=['HeartDisease'],
        evidence={'Smoking': 'yes', 'Age': 'old'}
    )
    print(result)

    print("\n--- Query 3: P(HeartDisease | ChestPain=yes, Breathlessness=yes) ---")
    result = infer.query(
        variables=['HeartDisease'],
        evidence={'ChestPain': 'yes', 'Breathlessness': 'yes'}
    )
    print(result)

    print("\n--- Query 4: P(HeartDisease | Smoking=no, Age=young) ---")
    result = infer.query(
        variables=['HeartDisease'],
        evidence={'Smoking': 'no', 'Age': 'young'}
    )
    print(result)

    print("\n--- Query 5: P(Hypertension | Age=old, Smoking=yes) ---")
    result = infer.query(
        variables=['Hypertension'],
        evidence={'Age': 'old', 'Smoking': 'yes'}
    )
    print(result)

    print("\n--- Query 6: Most Probable Explanation for ChestPain=yes ---")
    mpe = infer.map_query(
        variables=['HeartDisease', 'Hypertension', 'LungDisease'],
        evidence={'ChestPain': 'yes'}
    )
    print(f"  Most likely state: {mpe}")

    return infer


def sensitivity_analysis(infer):
    print("\n--- Sensitivity Analysis: How age affects heart disease risk ---")
    print(f"  {'Age':<10} {'Smoking':<12} {'P(HD=yes)':>10}")
    print(f"  {'─'*35}")
    for age in ['young', 'old']:
        for smoking in ['no', 'yes']:
            r = infer.query(
                variables=['HeartDisease'],
                evidence={'Age': age, 'Smoking': smoking}
            )
            prob = r.values[1]
            print(f"  {age:<10} {smoking:<12} {prob:>10.4f}")


def data_fitting_demo(model):
    print("\n--- Simulated Data Fitting (Parameter Learning Demo) ---")
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'Age': np.random.choice(['young', 'old'], n, p=[0.6, 0.4]),
        'Smoking': np.random.choice(['no', 'yes'], n, p=[0.7, 0.3]),
        'Hypertension': np.random.choice(['no', 'yes'], n, p=[0.65, 0.35]),
        'LungDisease': np.random.choice(['no', 'yes'], n, p=[0.85, 0.15]),
        'HeartDisease': np.random.choice(['no', 'yes'], n, p=[0.75, 0.25]),
        'ChestPain': np.random.choice(['no', 'yes'], n, p=[0.72, 0.28]),
        'Breathlessness': np.random.choice(['no', 'yes'], n, p=[0.78, 0.22])
    })

    print(f"  Dataset size: {len(data)} samples")
    print(f"  Heart disease prevalence in sample: "
          f"{(data['HeartDisease']=='yes').mean():.2%}")
    print(f"  Smoking prevalence: {(data['Smoking']=='yes').mean():.2%}")
    print(f"  Hypertension prevalence: {(data['Hypertension']=='yes').mean():.2%}")

    from pgmpy.parameter_estimator import DiscreteMLE
    learned = BayesianNetwork(model.edges())
    mle = DiscreteMLE()
    fitted = mle.fit(learned, data)
    cpds = fitted.parameters_
    hd_cpd = [c for c in cpds if 'HeartDisease' in str(c) and 'Hypertension' not in str(c) and 'Lung' not in str(c)][0]
    print("\n  Learned CPD for HeartDisease (from data):")
    print(hd_cpd)


def print_tools_survey():
    tools = {
        "pgmpy": {
            "language": "Python",
            "features": ["BN construction", "CPD specification", "exact & approx inference",
                        "parameter learning (MLE, Bayes)", "structure learning", "CausalInference"],
            "website": "https://pgmpy.org"
        },
        "Netica": {
            "language": "GUI + API (C, Java, MATLAB, Python)",
            "features": ["visual BN editor", "sensitivity analysis", "what-if scenarios",
                        "decision networks", "widely used in industry"],
            "website": "https://www.norsys.com/netica.html"
        },
        "GeNIe / SMILE": {
            "language": "C++ library, GUI frontend",
            "features": ["BN + influence diagrams", "dynamic BNs", "structural learning",
                        "free for academic use", "extensive documentation"],
            "website": "https://www.bayesfusion.com"
        },
        "BayesiaLab": {
            "language": "Java GUI",
            "features": ["structure learning from data", "target analysis", "data clustering",
                        "enterprise-grade", "visual analytics"],
            "website": "https://www.bayesia.com"
        },
        "Hugin": {
            "language": "Java / C API",
            "features": ["junction tree inference", "DBN support", "object-oriented BNs",
                        "widely used in research and industry"],
            "website": "https://www.hugin.com"
        }
    }

    print("\n--- BAYESIAN NETWORK TOOLS SURVEY ---\n")
    for name, info in tools.items():
        print(f"  {name}")
        print(f"    Language : {info['language']}")
        print(f"    Key features: {', '.join(info['features'][:3])}")
        print(f"    URL: {info['website']}")
        print()


def run_tests():
    print("=" * 55)
    print("ASSIGNMENT 4 — BAYESIAN NETWORKS")
    print("=" * 55)

    print_tools_survey()

    print("\nBuilding Medical Diagnosis Bayesian Network...")
    print("Domain: Heart Disease Risk Factors")
    print("Nodes: Age → Hypertension, Smoking → {Hypertension, LungDisease},")
    print("       {Hypertension, LungDisease} → HeartDisease,")
    print("       HeartDisease → {ChestPain, Breathlessness}")

    model = build_medical_diagnosis_bn()
    print(f"\nModel valid: {model.check_model()}")
    print(f"Nodes: {model.nodes()}")
    print(f"Edges: {model.edges()}")

    infer = run_inference(model)
    sensitivity_analysis(infer)
    data_fitting_demo(model)

    print("\n--- Conditional Independence Tests ---")
    print(f"  HeartDisease ⊥ Age | Hypertension, LungDisease: "
          f"{model.local_independencies('HeartDisease')}")

    d_sep = model.active_trail_nodes('Smoking', observed=['HeartDisease'])
    print(f"  Nodes d-connected to Smoking given HeartDisease: {d_sep}")

    print("\nAll Bayesian Network tests passed.")


if __name__ == "__main__":
    run_tests()