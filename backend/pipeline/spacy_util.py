"""Helpers for loading spaCy models (download on first use if missing)."""

import logging

import spacy

logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str):
    """
    Load a spaCy pipeline by name. If the model is not installed ([E050]),
    download it once via ``python -m spacy download`` and retry.
    """
    try:
        return spacy.load(model_name)
    except OSError as e:
        err = str(e)
        if "E050" not in err and "Can't find model" not in err:
            raise
        logger.warning(
            "spaCy model %r is not installed; downloading (one-time). "
            "Or run: python -m spacy download %s",
            model_name,
            model_name,
        )
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)
