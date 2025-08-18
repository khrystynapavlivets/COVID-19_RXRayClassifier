#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import load_img
os.getcwd()

# %%
class CovidDatasetLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
        self.data = {'image_path': [], 'label': []}

    def load_data(self):
        for class_name in self.classes:
            class_path = os.path.join(self.base_path, class_name, 'images')
            if not os.path.exists(class_path):
                print(f"Directory not found: {class_path}")
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data['image_path'].append(img_path)
                self.data['label'].append(class_name)

    def get_dataframe(self):
        return pd.DataFrame(self.data)

    def save_to_csv(self, csv_path):
        df = self.get_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

def load_covid_dataset():
    base_path = '/Users/admin/PycharmProjects/UdemyAI/PyTorchUltimate/Cnn_MulticlassClassification/data/COVID-19_Radiography_Dataset/'
    loader = CovidDatasetLoader(base_path)
    loader.load_data()
    df = loader.get_dataframe()
    return df, loader.classes



if __name__ == "__main__":
    df, classes = load_covid_dataset()
    print("Dataset Summary:")
    print(df.groupby('label').count())

# %%
# Function to display sample images
def display_samples(df, classes, samples_per_class=3):
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(classes):
        # Get 3 random samples for the class
        class_samples = df[df['label'] == class_name].sample(samples_per_class, random_state=42)
        
        for j, (_, row) in enumerate(class_samples.iterrows()):
            img = load_img(row['image_path'], target_size=(224, 224))
            plt.subplot(len(classes), samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"{class_name}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
#%%