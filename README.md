### MLE-Bench Agent — Multimodal AutoML Pipeline

This agent is designed to adapt across different ML competition modalities:
images, text, tabular, and sequence-to-sequence tasks.

It infers the problem type from metadata and dataset structure, selects the appropriate pipeline, and configures preprocessing, model architecture, and evaluation strategy without competition-specific hardcoding.

### How the Agent Understands Tasks?
- Inspects dataset schema + file types
- Maps the task to a pipeline:
- Image → CNN model
- Text → Transformer-based classification
- Tabular → Gradient-boosted models
- Seq2Seq → Encoder-decoder architecture
- Dynamically registers loss functions & metrics based on labels and competition rules

### Strategies Employed
- Automated hyperparameter selection
- Cross-validation and early stopping for stability

### To run the agent, need to specify the following environment variables :
The following example is for the "Aerial Cactus Identification" competition. The path to the dataset and other parameters should be modified as per the competition.

```bash
$env:DATA_PATH="C:\Users\Yashwant Raj\AppData\Local\mle-bench\data\aerial-cactus-identification\prepared\public"
$env:OUTPUT_PATH="submission.csv"
$env:COMPETITION_ID="aerial-cactus-identification"
```

### Once the environment variables are set, execute the agent with:
```bash
python my_agent_c.py
```

