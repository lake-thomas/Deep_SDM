# Grad-Cam implementation for NAIP imagery using PyTorch for Ailanthus Classification
# Thomas Lake, January 2026


# Imports
import os

conda_env = r"C:\Users\talake2\AppData\Local\anaconda3\envs\naip_ailanthus_env"
os.environ["GDAL_DATA"] = os.path.join(conda_env, "Library", "share", "gdal")
os.environ["PROJ_LIB"] = os.path.join(conda_env, "Library", "share", "proj")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env, "Library", "bin")

import torch # noqa: E402
from torch.utils.data import DataLoader # noqa: E402
from torchvision import transforms # noqa: E402
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import matplotlib.pyplot as plt # noqa: E402

from model import HostImageryClimateModel # noqa: E402
from datasets import HostNAIPDataset # noqa: E402
from train_utils import get_default_device, load_model_from_checkpoint # noqa: E402


# ------------------------------------------------------------
# Grad-CAM implementation
# ------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img, env):
        """
        img: Tensor (1, C, H, W)
        env: Tensor (1, E)
        """
        self.model.zero_grad()

        out = self.model(img, env)
        out.backward()

        # Grad-CAM computation
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

        return cam

# ------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------

def overlay_cam_on_rgb(rgb, cam, alpha=0.5):
    """
    rgb: (H, W, 3) in [0,1]
    cam: (h, w) resized later
    """
    cam = np.uint8(255 * cam)
    cam = plt.cm.jet(cam)[:, :, :3]

    cam = np.array(
        torch.nn.functional.interpolate(
            torch.tensor(cam).permute(2, 0, 1).unsqueeze(0),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    )

    overlay = alpha * cam + (1 - alpha) * rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay


def plot_gradcam(rgb, cam, prob, label, out_fp):
    overlay = overlay_cam_on_rgb(rgb, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nProb={prob:.2f}, Label={label}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()

# ------------------------------------------------------------
# Main Grad-CAM runner
# ------------------------------------------------------------

def run_gradcam():
    checkpoint_path = r"Y:\Ailanthus_NAIP_SDM\Outputs_NAIP_Ailanthus_Model_US_2026\outputs\ailanthus_us_full_uniform_image_climate_loc_jan162026\checkpoints\checkpoint_epoch_29.tar"
    csv_path = r"C:\Users\talake2\Desktop\Ailanthus_US_Uniform_PA_NAIP_256_jan26\Ailanthus_Pres_Bg_US_Uniform_Train_Val_Test_Dataset.csv"
    image_dir = r"C:\Users\talake2\Desktop\Ailanthus_US_Uniform_PA_NAIP_256_jan26"
    out_dir = r"Y:\Ailanthus_NAIP_SDM\Outputs_NAIP_Ailanthus_Model_US_2026\outputs\gradcam"

    device = get_default_device()

    df = pd.read_csv(csv_path)
    env_vars = [c for c in df.columns if c.startswith("wc2.1_30s") or c in ["ghm", "lat_norm", "lon_norm"]]

    model, _ = load_model_from_checkpoint(checkpoint_path, env_vars)
    model.to(device)
    model.eval()

    # Target layer for Grad-CAM
    target_layer = model.resnet.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    test_ds = HostNAIPDataset(
        csv_path,
        image_dir,
        split="test",
        environment_features=env_vars
    )

    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print("Running Grad-CAM on selected samples...")

    for i, batch in enumerate(test_dl):
        img, env, label, lat, lon, path = batch

        img = img.to(device)
        env = env.to(device)

        with torch.no_grad():
            prob = model(img, env).item()

        label_int = int(label.item())

        # Generate Grad-CAM
        cam = gradcam.generate(img, env)

        # Convert RGBN → RGB for visualization
        rgb = img[0, :3].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)

        fname = os.path.splitext(os.path.basename(path[0]))[0]
        out_fp = os.path.join(
            out_dir,
            f"{fname}_p{prob:.2f}_gradcam.png"
        )

        plot_gradcam(
            rgb,
            cam,
            prob,
            label_int,
            out_fp
        )

        # Optional cap on number of samples
        # if i >= 200:
        #     break

    print("Grad-CAM complete.")


if __name__ == "__main__":
    run_gradcam()















