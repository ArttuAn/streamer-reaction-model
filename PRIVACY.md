# Dataset and telemetry privacy

Generated dataset contents, face/gameplay images, training checkpoints, and
Weights & Biases run folders are not distributed in this branch. Existing local
copies remain ignored; prepare your own authorized dataset and use
`--dataset_dir` to select it. Review rights and consent before sharing facecam
material, labels, transcripts, embeddings, or models trained from that data.

Training sends no W&B telemetry by default. `--wandb` explicitly enables external
metrics and run metadata. `--wandb --wandb-artifacts` separately permits uploading
dataset identifiers, frame pairs/galleries, and model checkpoints. W&B may collect
machine/account metadata when opted in. Configure credentials through the
provider's protected WANDB_API_KEY environment variable; do not pass keys on the
command line or commit them. Review the W&B project's visibility before upload.

This commit removes generated artifacts from the current branch only. Earlier
commits, forks, clones, and existing W&B runs can still retain them. A targeted
history cleanup and deletion of unwanted external run artifacts are separate
steps. This patch does not contact W&B or delete remote runs.
