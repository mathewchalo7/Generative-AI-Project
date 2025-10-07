import numpy as np
import pandas as pd
from math import acos, degrees
# cspell:ignore recog - 

# --- CONFIGURATION / THRESHOLDS ---
SIM_STRONG = 0.95   # cosine similarity considered a strong match
SIM_WEAK = 0.85     # weak match region
CONTROL_NORM_THRESHOLD = 3.0  # acceptable process control limit

# --- REFERENCE VECTORS ---
ps  = np.array([0.02, 0.8, 58.5, 0.5])   # Perfect Spec
arc = np.array([0.05, 1.2, 57.8, 1.0])   # Acceptable Range Center
dpa = np.array([0.15, 2.8, 55.2, 2.5])   # Defect Pattern A (machining issue)
dpb = np.array([0.08, 1.5, 52.3, 1.2])   # Defect Pattern B (material issue)
cbs = np.array([0.07, 1.6, 56.5, 1.3])   # Current Batch Sample

# Tolerances for each component (dimension, surface finish, hardness, weight)
tolerances = np.array([0.05, 0.6, 1.5, 0.8])

# Process control factor vectors
mach = np.array([1.0, 0.7, 0.3, 0.2])
mat  = np.array([0.2, 0.4, 1.0, 0.8])
tool = np.array([0.6, 1.0, 0.5, 0.3])

# --- HELPER FUNCTIONS ---
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na != 0 and nb != 0 else 0.0

def classify_defect(cmp_vec, patterns, labels, sim_strong=SIM_STRONG, sim_weak=SIM_WEAK):
    sims = [cosine_similarity(cmp_vec, p) for p in patterns]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    label = labels[best_idx]
    if best_sim >= sim_strong:
        confidence = "high"
    elif best_sim >= sim_weak:
        confidence = "medium"
    else:
        if cosine_similarity(cmp_vec, arc) > sim_weak:
            label = "acceptable"
            confidence = "medium"
        else:
            label = "acceptable"
            confidence = "low"
    return {
        "label": label,
        "best_similarity": best_sim,
        "confidence": confidence,
        "all_similarities": sims
    }

def deviation_vector(current, target):
    return current - target

def projection_onto(u, v):
    u_norm_sq = np.dot(u, u)
    if u_norm_sq == 0:
        return np.zeros_like(v)
    scalar = np.dot(v, u) / u_norm_sq
    return scalar * u

def perp_component(u, v):
    return v - projection_onto(u, v)

def angle_between(a, b):
    sim = cosine_similarity(a, b)
    sim = max(min(sim, 1.0), -1.0)  # numerical safety
    return degrees(acos(sim))

# Principal defect directions
dpa_u = normalize(dpa - ps)
dpb_u = normalize(dpb - ps)

# --- MAIN QUALITY CONTROL CLASS ---
class QualityControl:
    def __init__(self, perfect_spec, acceptable_center, defect_patterns, defect_labels,
                 process_factors, tolerances):
        self.ps = perfect_spec
        self.arc = acceptable_center
        self.patterns = defect_patterns
        self.labels = defect_labels
        self.factors = process_factors
        self.tolerances = tolerances

    def defect_recognition(self, sample):
        result = classify_defect(sample, self.patterns + [self.arc], self.labels + ["acceptable"])
        sig = np.sign(sample - self.ps)
        signature = ["below" if s == -1 else ("equal" if s == 0 else "above") for s in sig]
        result["signature"] = signature
        return result

    def specification_deviation(self, sample):
        dev = deviation_vector(sample, self.ps)
        magnitude = np.linalg.norm(dev)
        proj_on_ps = projection_onto(self.ps, sample)
        perp = perp_component(self.ps, sample)
        perp_mag = np.linalg.norm(perp)
        comp_exceed = list(np.abs(dev) > self.tolerances)
        components = [
            {
                "index": i,
                "dev": float(dev[i]),
                "tolerance": float(self.tolerances[i]),
                "exceeds": comp_exceed[i]
            } for i in range(len(dev))
        ]
        return {
            "deviation_vector": dev,
            "deviation_magnitude": magnitude,
            "projection_on_ps": proj_on_ps,
            "perpendicular_component": perp,
            "perpendicular_magnitude": perp_mag,
            "components": components
        }

    def process_projections(self, sample):
        projections = {}
        for name, vec in self.factors.items():
            proj = projection_onto(vec, sample - self.ps)
            proj_mag = np.linalg.norm(proj)
            projections[name] = {
                "projection_vector": proj,
                "projection_magnitude": proj_mag
            }
        largest = max(projections.items(), key=lambda kv: kv[1]["projection_magnitude"])
        return {
            "projections": projections,
            "root_cause": largest[0],
            "root_magnitude": largest[1]["projection_magnitude"]
        }

    def corrective_action(self, sample):
        correction = self.ps - sample
        magnitude = np.linalg.norm(correction)
        angle = angle_between(sample - self.ps, self.ps)
        priorities = sorted(
            [
                {"index": i, "required_correction": float(correction[i]), "abs": abs(correction[i])}
                for i in range(len(correction))
            ],
            key=lambda x: x["abs"],
            reverse=True
        )
        return {
            "correction_vector": correction,
            "magnitude": magnitude,
            "angle_degrees": angle,
            "priorities": priorities
        }

    def statistical_process_control(self, sample):
        dev = sample - self.arc
        dev_norm = np.linalg.norm(dev)
        within_control = dev_norm <= CONTROL_NORM_THRESHOLD
        proj_dpa = np.dot(dev, dpa_u)
        proj_dpb = np.dot(dev, dpb_u)
        alerts = []
        if not within_control:
            alerts.append({
                "type": "out_of_control",
                "message": f"Deviation norm {dev_norm:.3f} exceeds control threshold {CONTROL_NORM_THRESHOLD}."
            })
        for i, (d, tol) in enumerate(zip(np.abs(sample - self.ps), self.tolerances)):
            if d > tol:
                alerts.append({
                    "type": "component_exceed",
                    "index": i,
                    "value": float(d),
                    "tolerance": float(tol)
                })
        return {
            "deviation_from_arc": dev,
            "deviation_norm": dev_norm,
            "within_control": within_control,
            "proj_on_dpa": float(proj_dpa),
            "proj_on_dpb": float(proj_dpb),
            "alerts": alerts
        }

# --- RUN THE FULL ANALYSIS ---
qc = QualityControl(
    perfect_spec=ps,
    acceptable_center=arc,
    defect_patterns=[dpa, dpb],
    defect_labels=["defect A", "defect B"],
    process_factors={
        "machining_precision": mach,
        "material_quality": mat,
        "tool_wear": tool
    },
    tolerances=tolerances
)

# Execute all modules on the current batch
recog = qc.defect_recognition(cbs)
spec_dev = qc.specification_deviation(cbs)
proc_proj = qc.process_projections(cbs)
correction = qc.corrective_action(cbs)
spc = qc.statistical_process_control(cbs)

# --- SUMMARY OUTPUT ---
summary = {
    "Defect Recognition": recog,
    "Deviation Magnitude": spec_dev["deviation_magnitude"],
    "Root Cause": proc_proj["root_cause"],
    "Root Cause Magnitude": proc_proj["root_magnitude"],
    "Required Correction Magnitude": correction["magnitude"],
    "Within Control": spc["within_control"],
    "SPC Alerts": spc["alerts"]
}

import pprint
pprint.pprint(summary)


