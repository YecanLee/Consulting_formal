# Define base and novel class IDs for ceiling painting dataset
# Classes: ['#', 'brief', 'mural', 'relief'] - indices 0, 1, 2, 3

# For zero-shot/open-vocabulary setup:
# - Base classes: seen during training
# - Novel classes: only seen during testing

# Example split:
CEILING_BASE_IDS = [0, 1, 2]  # mural, brief, mural - seen during training
CEILING_NOVEL_IDS = [3]  # relief - only for zero-shot testing

# Train on all classes:
CEILING_ALL_IDS = [0, 1, 2, 3]  # All classes for standard training 
