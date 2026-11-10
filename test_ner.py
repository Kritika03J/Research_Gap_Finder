
from multiple_paper_ner import process_paper_pages

pages = [
    {
        "page": 1,
        "text": """
        Deep learning methods are widely used for image
        classification. Convolutional neural networks are
        evaluated on the ImageNet dataset for image
        recognition tasks.
        """
    },
    {
        "page": 2,
        "text": """
        The proposed method uses transfer learning.
        The experiments demonstrate improved classification
        performance on the dataset.
        """
    }
]

ner_df, rel_df, stats = process_paper_pages(pages)

print("\n========== ENTITIES ==========")
print(ner_df)

print("\n========== RELATIONS ==========")
print(rel_df)

print("\n========== STATISTICS ==========")
print(stats)

ner_df.to_csv(
    "test_entities.csv",
    index=False
)

rel_df.to_csv(
    "test_relations.csv",
    index=False
)
