import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class AspectPrediction:
    label: str
    score: float
    sentiment: Optional["SentimentPrediction"] = None


@dataclass
class SentimentPrediction:
    label: str
    score: float


class ABSAService:
    """
    Loads the fine-tuned Hugging Face models exported from Colab and exposes
    helper methods that are easy to call from the Streamlit UI.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        aspect_threshold: float = 0.3,
    ) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.aspect_threshold = aspect_threshold

        models_root = self.base_dir / "models"
        self.aspect_dir = models_root / "aspect"
        self.sentiment_dir = models_root / "sentiment"

        if not self.aspect_dir.exists():
            raise FileNotFoundError(f"Aspect model not found at {self.aspect_dir}")
        if not self.sentiment_dir.exists():
            raise FileNotFoundError(
                f"Sentiment model not found at {self.sentiment_dir}"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.aspect_tokenizer = AutoTokenizer.from_pretrained(self.aspect_dir)
        self.aspect_model = (
            AutoModelForSequenceClassification.from_pretrained(self.aspect_dir)
            .to(self.device)
            .eval()
        )

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_dir)
        self.sentiment_model = (
            AutoModelForSequenceClassification.from_pretrained(self.sentiment_dir)
            .to(self.device)
            .eval()
        )

        # Read id2label mapping directly from config to keep names in sync
        self.aspect_labels = self._read_labels(self.aspect_dir)
        self.sentiment_labels = self._read_labels(self.sentiment_dir)

    @staticmethod
    def _read_labels(model_dir: Path) -> Dict[int, str]:
        custom_labels_path = model_dir / "labels.json"
        if custom_labels_path.exists():
            with custom_labels_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {idx: label for idx, label in enumerate(data)}
            if isinstance(data, dict):
                try:
                    return {int(idx): label for idx, label in data.items()}
                except ValueError:
                    # keys might be string labels -> invert order preserving
                    return {
                        idx: label for idx, label in enumerate(data.values())
                    }

        config_path = model_dir / "config.json"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        id2label = data.get("id2label", {})
        return {int(idx): label for idx, label in id2label.items()}

    def update_threshold(self, threshold: float) -> None:
        self.aspect_threshold = threshold

    def predict_aspects(self, text: str) -> List[AspectPrediction]:
        encoded = self.aspect_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.aspect_model(**encoded).logits

        scores = torch.sigmoid(logits).cpu().numpy()[0]

        predictions: List[AspectPrediction] = []
        for idx, score in enumerate(scores):
            if score >= self.aspect_threshold:
                label = self.aspect_labels.get(idx, f"LABEL_{idx}")
                predictions.append(AspectPrediction(label=label, score=float(score)))

        predictions.sort(key=lambda p: p.score, reverse=True)
        return predictions

    def predict_sentiment(
        self,
        text: str,
        aspect: Optional[str] = None,
    ) -> SentimentPrediction:
        if aspect:
            enriched_text = f"aspect: {aspect} text: {text}"
        else:
            enriched_text = text

        encoded = self.sentiment_tokenizer(
            enriched_text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.sentiment_model(**encoded).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        label = self.sentiment_labels.get(top_idx, f"LABEL_{top_idx}")
        return SentimentPrediction(label=label, score=float(probs[top_idx]))

    def analyze_text(self, text: str) -> Dict[str, object]:
        aspects = self.predict_aspects(text)
        global_sentiment = self.predict_sentiment(text)

        enriched_aspects: List[AspectPrediction] = []
        for aspect in aspects:
            sentiment = self.predict_sentiment(text, aspect.label)
            enriched_aspects.append(
                AspectPrediction(
                    label=aspect.label,
                    score=aspect.score,
                    sentiment=sentiment,
                )
            )

        if enriched_aspects:
            aspect_sentiment = self.aggregate_sentiment(enriched_aspects)
            # Prefer aspect-aware sentiment only when it is at least as confident as the global one.
            sentiment = (
                aspect_sentiment
                if aspect_sentiment.score >= global_sentiment.score
                else global_sentiment
            )
        else:
            sentiment = global_sentiment
        return {
            "text": text,
            "aspects": enriched_aspects,
            "sentiment": sentiment,
        }

    def aggregate_sentiment(
        self, aspect_predictions: List[AspectPrediction]
    ) -> SentimentPrediction:
        """
        Tổng hợp sentiment tổng thể dựa trên aspect có confidence cao nhất.
        Ưu tiên NEG > POS > NEU nếu độ tin cậy tương đương.
        """

        if not aspect_predictions:
            return SentimentPrediction(label="NEU", score=0.0)

        scored_items: List[tuple[float, SentimentPrediction]] = []
        for item in aspect_predictions:
            if not item.sentiment:
                continue
            weight = item.score * item.sentiment.score
            scored_items.append((weight, item.sentiment))

        if not scored_items:
            return SentimentPrediction(label="NEU", score=0.0)

        # Sort by weight desc, with priority NEG > POS > NEU for equal weights
        priority = {"NEG": 0, "NEGATIVE": 0, "POS": 1, "POSITIVE": 1, "NEU": 2, "NEUTRAL": 2}
        scored_items.sort(
            key=lambda x: (-x[0], priority.get(x[1].label.upper(), 3))
        )

        top_weight, top_sentiment = scored_items[0]
        return SentimentPrediction(label=top_sentiment.label, score=top_sentiment.score)


def get_service(aspect_threshold: float = 0.3) -> ABSAService:
    """
    Helper for Streamlit `st.cache_resource` usage.
    """

    return ABSAService(aspect_threshold=aspect_threshold)

