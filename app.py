import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class PhotoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SNEAKERS CHECKER")
        self.root.geometry("400x400")
        self.root.config(bg="#f0f0f0")

        self.label = tk.Label(root, text="Загрузите фото", font=("Helvetica", 16), bg="#f0f0f0")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="Загрузить фото", command=self.upload_photo, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(root, bg="#f0f0f0")
        self.image_label.pack(pady=20)

        self.check_button = tk.Button(root, text="Проверить", command=self.check_photo_PLACEHOLDER, bg="#2196F3", fg="white", font=("Helvetica", 12))
        self.check_button.pack(pady=10)

        self.image_path = None

    def upload_photo(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.display_image()

    def display_image(self):
        img = Image.open(self.image_path)
        img = img.resize((200, 200), Image.LANCZOS) 
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def check_photo_PLACEHOLDER(self):
        if self.image_path:
            messagebox.showinfo("инфа", "фотка загружена успешно")
        else:
            messagebox.showwarning("предупреждение", "пж, загрузите фото перед проверкой")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.mainloop()