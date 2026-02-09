import pandas as pd
import logging
import re


class Preprocess_News_Papers():
    """Preprocesses newspaper articles by cleaning text and segmenting into overlapping windows."""

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, sep=";")

    def clean_data(self, txt):
        """Remove special characters and URLs from text."""
        txt = re.sub(r'[^\w\s.,;:!?-]', '', txt)
        url_pattern = r'https?://\S+|www\.\S+'
        txt = re.sub(url_pattern, '', txt)

        txt = txt.strip()
        return txt

    def splitting_text(self, text, max_length, min_words, overlap, published_at):
        """Split text into overlapping segments of fixed word length.

        Args:
            text: Raw article text.
            max_length: Maximum number of words per segment.
            min_words: Minimum word count to trigger segmentation.
            overlap: Overlap ratio between consecutive segments (0 to 1).
            published_at: Publication year of the article.

        Returns:
            List of (segment_text, segment_id, published_at) tuples.
        """
        logging.info("Text Cleaned")

        words = text.split()
        if len(words) < min_words:
            return [(text, 0, published_at)]

        overlap_length = int(max_length * overlap)
        segments = []
        start = 0
        segment_id = 0

        while start < len(words):
            end = min(start + max_length, len(words))
            segment = ' '.join(words[start:end])

            if len(segment.split()) > max_length:
                segment = ' '.join(words[start:start+max_length])

            segments.append((segment, segment_id, published_at))

            start += max_length - overlap_length
            segment_id += 1

        return segments

    def apply_splitting(self, max_length, min_words, overlap):
        """Apply segmentation to all articles in the DataFrame.

        Args:
            max_length: Maximum number of words per segment.
            min_words: Minimum word count to trigger segmentation.
            overlap: Overlap ratio between consecutive segments.

        Returns:
            DataFrame with columns: article_id, segment_id, segment_text, annee.
        """
        all_segments = []
        for idx, row in self.df[0:1001].iterrows():
            segments = self.splitting_text(
                text=row['TXT'],
                max_length=max_length,
                min_words=min_words,
                overlap=overlap,
                published_at=row["ANNEE"]
            )

            for segment_text, segment_id, annee in segments:
                all_segments.append({
                    "article_id": idx,
                    "segment_id": segment_id,
                    "segment_text": segment_text,
                    "annee": annee
                })

        logging.info("Segmentation Done !")
        segmented_df = pd.DataFrame(all_segments)
        return segmented_df

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    csv_path = "raw_articles.csv"
    
    preprocessor = Preprocess_News_Papers(csv_path)
    result_df = preprocessor.apply_splitting(
        max_length=200,
        min_words=100,
        overlap=0.2
    )
    
    result_df.to_csv("Spatial_Annotation_Detection/data/df_sample.csv", index=False, sep=";")