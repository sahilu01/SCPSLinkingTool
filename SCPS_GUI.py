import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from tkinter import Label, Entry, Button, Text, Scrollbar, filedialog, messagebox
import webbrowser


class STOCPredictionApp:
    def __init__(self):
        self.stoc_df, self.scps_df, self.learning_data = None, None, None
        self.root = tk.Tk()
        self.root.title("STOC Prediction GUI")

        self.create_widgets()

    def load_csv(self):
        stoc_file_path = filedialog.askopenfilename(title="Select STOC CSV file",
                                                    filetypes=[("CSV files", "*.csv")])
        scps_file_path = filedialog.askopenfilename(title="Select SCPS CSV file",
                                                    filetypes=[("CSV files", "*.csv")])

        if stoc_file_path and scps_file_path:
            self.stoc_df = pd.read_csv(stoc_file_path)
            self.scps_df = pd.read_csv(scps_file_path)

            merged_df = pd.merge(self.scps_df, self.stoc_df, how='left',
                                 left_on='Custom field (Link to Other Tickets)',
                                 right_on='Issue key')

            self.learning_data = merged_df[
                ['Issue key_x', 'Issue key_y', 'Summary_x', 'Summary_y', 'Description_y']]
            self.learning_data.columns = ['SCPS', 'STOC', 'SCPS_Summary', 'STOC_Summary',
                                          'STOC_Description']
            self.learning_data = self.learning_data.dropna(subset=['STOC'])

            success_label = Label(self.root, text="CSV files loaded successfully!", fg="green")
            success_label.pack(pady=10)

    def predict_stoc(self):
        if self.stoc_df is None or self.scps_df is None:
            messagebox.showerror("Error", "Please load the CSV files first.")
            return

        user_input_text = user_input_entry.get()
        user_input_str = ' '.join(user_input_text.split())

        try:
            X = self.learning_data[['SCPS_Summary', 'STOC_Summary', 'STOC_Description']]
            y = self.learning_data['STOC']

            model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('classifier', MLPClassifier())
            ])

            model.fit(X.apply(lambda x: ' '.join(map(str, x)), axis=1), y)

            user_pred_prob = model.predict_proba([user_input_str])[0]
            user_pred_classes = model.classes_
            sorted_indices = user_pred_prob.argsort()[::-1]

            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"User Input: {user_input_text}\nTop 10 Predictions:\n")

            for i in range(10):
                index = sorted_indices[i]
                stoc_url = f"https://gdncomm.atlassian.net/browse/{user_pred_classes[index]}"
                stoc_summary = self.learning_data.loc[
                    self.learning_data['STOC'] == user_pred_classes[index], 'STOC_Summary'].values[
                    0]

                result_text.insert(tk.END,
                                   f"\nPrediction: {user_pred_classes[index]}, Probability: {user_pred_prob[index]:.4f}, URL: ")
                result_text.insert(tk.END, stoc_url, f"link_{index}")
                result_text.insert(tk.END, f"\nSTOC Summary: {stoc_summary}\n")

                result_text.tag_configure(f"link_{index}", foreground="blue", underline=True)
                result_text.tag_bind(f"link_{index}", "<Button-1>",
                                     lambda event, url=stoc_url: webbrowser.open(url))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def create_widgets(self):
        Label(self.root, text="Load CSV files:").pack(pady=10)
        load_csv_button = Button(self.root, text="Load CSV", command=self.load_csv)
        load_csv_button.pack(pady=10)

        Label(self.root, text="Enter your text:").pack(pady=10)
        global user_input_entry
        user_input_entry = Entry(self.root, width=50)
        user_input_entry.pack(pady=10)

        predict_button = Button(self.root, text="Predict STOC", command=self.predict_stoc)
        predict_button.pack(pady=10)

        global result_text
        result_text = Text(self.root, height=20, width=80, wrap=tk.WORD)
        result_text.pack(pady=10)

        scrollbar = Scrollbar(self.root, command=result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        result_text.config(yscrollcommand=scrollbar.set)

        self.root.mainloop()


# Run the application
if __name__ == "__main__":
    app = STOCPredictionApp()
