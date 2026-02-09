from Preprocessing.Preprocess_News_Papers import Preprocess_News_Papers
import logging

def execute():
    csv_path = "raw_articles.csv"
    preprocessor = Preprocess_News_Papers(csv_path)
    
    segmented_df = preprocessor.apply_splitting(
        max_length=300,
        min_words=32,
        overlap=0.1
    )
    
    segmented_df.to_csv('Spatial_Annotation_Detection/data/df_sample.csv', index=False)
    logging.info("Textual News Paper Preprocessed -- Segmented")
    
if __name__ == "__main__":
    execute()
