# Literature Matrix

| Method/dataset | Role in paper | Why it matters |
|---|---|---|
| MVTec AD | Main benchmark | Standard industrial anomaly detection benchmark. |
| VisA | Main benchmark | More diverse visual anomaly categories and masks. |
| MPDD | Main benchmark | Metal/industrial parts with reflective and local defects. |
| MVTec AD 2 | Strengthening benchmark | Harder conditions: lighting shift, tiny defects, transparent/overlapping objects. |
| PatchCore | Memory/prototype baseline | Strong non-diffusion localization baseline. |
| PaDiM | Memory/statistical baseline | Classic normal feature modeling baseline. |
| SimpleNet | Efficient AD baseline | Direct challenge to Lite-SEER-AD's efficiency story. |
| DRAEM | Reconstruction/synthetic baseline | Synthetic anomaly training comparison. |
| RD4AD | Distillation baseline | Strong industrial anomaly detection reference. |
| UniAD | Unified transformer baseline | Multi-class unified anomaly detection comparison. |
| DiffusionAD | Diffusion nearest baseline | Direct diffusion reconstruction comparison. |
| DDAD | Diffusion nearest baseline | Direct denoising diffusion AD comparison. |
| InvAD | Optional diffusion baseline | Recent diffusion-style comparator for 2025+ reviewer expectations. |

## Paper Narrative

The proposed paper should avoid claiming that diffusion reconstruction alone is novel. The defensible claim is that repair becomes part of anomaly reasoning:

1. Residuals nominate suspicious regions.
2. HN-SEV suppresses normal high-residual regions.
3. Local diffusion repair creates a counterfactual version of each ROI.
4. CRV uses score drop as verification evidence.
5. LC-RDS makes the repair budget explicit and measurable.
