import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Classifier")
        self.root.geometry("400x300")  # Set window size

        # Load the trained model
        self.model = load_model('cats_vs_dogs_model.h5')

        # Create widgets
        self.title_label = ttk.Label(root, text="Cat vs Dog Image Classifier", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        self.upload_button = ttk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.result_label = ttk.Label(root, text="Prediction:", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        self.result_text = tk.Text(root, height=4, width=40, wrap=tk.WORD)
        self.result_text.pack(pady=10)

    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Preprocess and predict
            img = image.load_img(file_path, target_size=(128, 128))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Predict
            prediction = self.model.predict(img)
            if prediction > 0.5:
                result = "Dog"
            else:
                result = "Cat"

            # Display result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Prediction: {result}")

# Initialize the Tkinter window
root = tk.Tk()
app = ImageClassifierApp(root)

# Start the Tkinter event loop
root.mainloop()
