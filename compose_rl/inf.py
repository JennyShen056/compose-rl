import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class LlamaRewardHead(nn.Module):
    """Reward head for classification, similar to LlamaRewardHead in provided code"""

    def __init__(self, config, n_labels=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.out_proj = nn.Linear(config.hidden_size, n_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class LlamaForSequenceClassification(nn.Module):
    """Llama model with a classification head on top"""

    def __init__(self, base_model, config):
        super().__init__()
        self.model = base_model
        self.reward_head = LlamaRewardHead(config, n_labels=1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]

        # For classification, use the last token's hidden state
        # We can either use the [EOS] token or the last non-padding token
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            last_token_hidden = hidden_states[batch_indices, last_token_indices]
        else:
            # Use the last token
            last_token_hidden = hidden_states[:, -1]

        # Apply reward head
        logits = self.reward_head(last_token_hidden)

        return logits


class RewardModelEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """Load the trained reward model from checkpoint"""
        print(f"Loading model from {self.model_path}")

        # Load config
        config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
        )

        # Create the classification model
        model = LlamaForSequenceClassification(base_model, config)

        # Load all checkpoint shards
        model_dirs = os.listdir(self.model_path)
        checkpoint_files = [f for f in model_dirs if f.endswith(".distcp")]

        state_dict = {}
        for file in checkpoint_files:
            checkpoint = torch.load(f"{self.model_path}/{file}", map_location="cpu")
            state_dict.update(checkpoint)

        # Extract and load base model weights
        base_model_weights = {
            k.replace("reward_base_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("reward_base_model.") or k.startswith("model.")
        }
        base_model_state = {
            k: v for k, v in base_model_weights.items() if k in base_model.state_dict()
        }
        base_model.load_state_dict(base_model_state, strict=False)

        # Extract and load reward head weights
        reward_head_weights = {
            k.replace("reward_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("reward_head.")
            or k.startswith("classifier.")
            or k.startswith("score.")
        }

        if reward_head_weights:
            reward_head_state = {
                k: v
                for k, v in reward_head_weights.items()
                if k in model.reward_head.state_dict()
            }
            model.reward_head.load_state_dict(reward_head_state, strict=False)
        else:
            print("Warning: No reward head weights found in checkpoint")

        return model

    def process_batch(self, texts):
        """Process a batch of texts for model input"""
        inputs = []
        for text in texts:
            message = [
                {
                    "role": "user",
                    "content": f"categorize the movie review into positive (1) or negative (0): {text}",
                }
            ]

            # Apply the chat template to format the input
            inputs.append(
                self.tokenizer.apply_chat_template(message, return_tensors=None)
            )

        # Tokenize and create batch
        encoded_inputs = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        return encoded_inputs

    def predict(self, inputs):
        """Run prediction on processed inputs"""
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs)

            # For binary classification, convert logits to probabilities with sigmoid
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int()

            return predictions.cpu(), probs.cpu()

    def evaluate_dataset(self, dataset, batch_size=8):
        """Evaluate the model on a dataset"""
        all_predictions = []
        all_labels = []
        all_probs = []

        # Create batches
        num_samples = len(dataset)
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch = dataset[i : min(i + batch_size, num_samples)]

            texts = [example["text"] for example in batch]
            labels = [example["label"] for example in batch]

            inputs = self.process_batch(texts)
            predictions, probs = self.predict(inputs)

            all_predictions.extend(predictions.numpy().flatten())
            all_labels.extend(labels)
            all_probs.extend(probs.numpy().flatten())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_predictions,
            "probabilities": all_probs,
            "labels": all_labels,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reward model on IMDB dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/reward_model/ep1-ba2541",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = "cpu"

    # Load IMDB dataset
    print("Loading IMDB test dataset...")
    dataset = load_dataset("imdb", split="test")

    # Initialize evaluator
    evaluator = RewardModelEvaluator(args.model_path, device=args.device)

    # Evaluate model
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(dataset, batch_size=args.batch_size)

    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    np.save(
        "results/evaluation_metrics.npy",
        {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
        },
    )

    print("Evaluation complete! Results saved to results/evaluation_metrics.npy")


if __name__ == "__main__":
    main()
