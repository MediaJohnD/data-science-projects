from typing import Iterable, List, Dict


def link_prediction(predictions: Iterable[int], ids: Iterable[int]) -> List[Dict]:
    """Attach predictions to record identifiers."""
    return [
        {"id": i, "prediction": int(pred)}
        for i, pred in zip(ids, predictions)
    ]
