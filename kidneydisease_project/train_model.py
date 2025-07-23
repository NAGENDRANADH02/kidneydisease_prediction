 

# 1) Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

 
class CKDModelTrainer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.models = {}
        self.accuracies = {}
        self.label_encoder = LabelEncoder()

         
        self.features_to_use = [
            'specific_gravity', 'red_blood_cell_count', 'hemoglobin', 'hypertension',
            'diabetesmellitus', 'albumin', 'packed_cell_volume', 'appetite', 'sodium',
            'pus_cell', 'pedal_edema', 'blood_urea', 'blood glucose random', 'anemia',
            'sugar', 'blood_pressure', 'serum_creatinine', 'red_blood_cells',
            'pus_cell_clumps', 'coronary_artery_disease', 'age', 'white_blood_cell_count'
        ]

    def load_data(self):
        print("📥 Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print("✅ Dataset shape:", self.df.shape)

    def preprocess(self):
        print("🧹 Preprocessing...")
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.label_encoder.fit_transform(self.df[col])
        print("✅ Preprocessing complete.")

    def split_data(self):
        X = self.df[self.features_to_use]
        y = self.df['class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=21
        )
        print(f"🧪 Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    # Individual model methods
    def train_knn(self):
        print("🔷 Training KNN...")
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, pred)
        self.models['KNN'] = model
        self.accuracies['KNN'] = acc
        print(f"✅ KNN Accuracy: {acc * 100:.2f}%")

    def train_naive_bayes(self):
        print("🟠 Training Naive Bayes...")
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, pred)
        self.models['Naive Bayes'] = model
        self.accuracies['Naive Bayes'] = acc
        print(f"✅ Naive Bayes Accuracy: {acc * 100:.2f}%")

    def train_logistic_regression(self):
        print("🟢 Training Logistic Regression...")
        model = LogisticRegression(solver='liblinear')
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, pred)
        self.models['Logistic Regression'] = model
        self.accuracies['Logistic Regression'] = acc
        print(f"✅ Logistic Regression Accuracy: {acc * 100:.2f}%")

    def show_accuracies(self):
        print("\n📊 Accuracy Summary:")
        for name, acc in self.accuracies.items():
            print(f"{name}: {acc * 100:.2f}%")

        # Plot accuracies
        plt.bar(self.accuracies.keys(), [v * 100 for v in self.accuracies.values()],
                color=['#3498db', '#e67e22', '#2ecc71'])
        plt.xlabel("Algorithms")
        plt.ylabel("Accuracy (%)")
        plt.title("Model Comparison")
        plt.show()

    def save_best_model(self):
        best_name = max(self.accuracies, key=self.accuracies.get)
        best_model = self.models[best_name]
        joblib.dump(best_model, 'predictor/CKD.pkl')
        print(f"\n💾 Best Model: {best_name} saved as 'predictor/CKD.pkl'")


# Run the training pipeline
if __name__ == "__main__":
    trainer = CKDModelTrainer("modified_dataset.csv")
    trainer.load_data()
    trainer.preprocess()
    trainer.split_data()

    trainer.train_knn()
    trainer.train_naive_bayes()
    trainer.train_logistic_regression()

    trainer.show_accuracies()
    trainer.save_best_model()
