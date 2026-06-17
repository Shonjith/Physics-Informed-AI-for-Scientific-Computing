import numpy as np
import torch
import matplotlib.pyplot as plt

from model import DualPINN
from data import norm_x

def evaluate_model():

    # Device
    DEVICE = "cpu"

    x_min = 0.0
    x_max = 10.0

    # Model
    model = DualPINN().to(DEVICE)

    try:

        model.load_state_dict(torch.load("pinn_model.pt",map_location=DEVICE))

        model.eval()

    except FileNotFoundError:

        print("Error: 'pinn_model.pt' not found.")

        print("Run training first.")

        exit()

    try:

        gt = np.load("ground_truth_defect.npz")

        x_gt = gt['x_grid']

        S_gt = gt['S_true']

        has_gt = True

    except FileNotFoundError:

        has_gt = False

    x_eval = np.linspace(x_min,x_max,1000)

    x_eval_n = torch.tensor(norm_x(x_eval).astype(np.float32)[:, None])

    with torch.no_grad():

        S_pred = (model.forward_S(x_eval_n).numpy().flatten())

    print("\n" + "=" * 60)

    print(
        "PINN INVERSE "
        "PROBLEM EVALUATION"
    )

    print("=" * 60)

    threshold = 0.1

    defect_indices = np.where(S_pred > threshold)[0]

    if len(defect_indices) > 0:

        x_start = x_eval[defect_indices[0]]

        x_end = x_eval[defect_indices[-1]]

        peak_val = np.max(S_pred)

        peak_loc = x_eval[np.argmax(S_pred)]

        print("[DISCOVERED DEFECT DETECTED]")

        print(f"Estimated Range: x = {x_start:.2f} to {x_end:.2f}")

        print(f"Peak Intensity: S = {peak_val:.4f} at x = {peak_loc:.2f}")

    else:

        print("[NO SIGNIFICANT DEFECT DETECTED]")

    print("-" * 60)

    print("Spatial Profile (Sampled Points):")

    print( f"{'x-coord':>10} | {'Predicted S(x)':>15}")

    print("-" * 30)

    for i in range(0,len(x_eval),100):

        print(f"{x_eval[i]:>10.2f} | {S_pred[i]:>15.4f}")

    peak_idx = np.argmax(S_pred)

    if (peak_idx % 100 != 0 and 'peak_val' in locals() and peak_val > threshold):

        print(f"{x_eval[peak_idx]:>10.2f} | {S_pred[peak_idx]:>15.4f} <-- PEAK")
    print("=" * 60)

    print("Generating visualization...")

    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(10, 5))

    # Prediction
ax.fill_between(x_eval,S_pred,alpha=0.3,color='salmon')

ax.plot(x_eval,S_pred,color='salmon',lw=3,label='PINN Predicted S(x)')

# Ground truth
if has_gt:

    ax.plot(x_gt,S_gt,color='mediumseagreen',lw=2,ls='--',label='True Defect S(x)')

    ax.axvspan(3.0,4.5,alpha=0.1,color='mediumseagreen')

ax.set_title('Inverse PINN: Discovered Spatial Defect S(x)',fontsize=14,fontweight='bold',pad=15)

ax.set_xlabel('Spatial Coordinate (x)',fontsize=12)

ax.set_ylabel('Defect Strength S(x)',fontsize=12)

ax.set_xlim(0, 10)

ax.set_ylim(-0.2,max(3.0, peak_val + 0.5)if 'peak_val' in locals()else 3.0)

ax.grid(True, alpha=0.2)

ax.legend(loc='upper right',fontsize=10)

save_path = 'final_defect_plot.png'

plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.close()

print(f"Image successfully saved to: {save_path}")
print("=" * 60 + "\n")