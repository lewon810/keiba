import customtkinter as ctk
import scraper
import predictor
import threading

class KeibaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Keiba Prediction App")
        self.geometry("800x600")

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(self, text="Keiba Prediction", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Input Frame
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.url_label = ctk.CTkLabel(self.input_frame, text="Netkeiba URL:")
        self.url_label.pack(side="left", padx=10)
        
        self.url_entry = ctk.CTkEntry(self.input_frame, width=400, placeholder_text="Paste netkeiba shutuba table URL here")
        self.url_entry.pack(side="left", padx=10, fill="x", expand=True)

        self.predict_button = ctk.CTkButton(self.input_frame, text="Predict", command=self.start_prediction)
        self.predict_button.pack(side="right", padx=10)

        # Output Area
        self.output_textbox = ctk.CTkTextbox(self, width=760, height=400)
        self.output_textbox.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.output_textbox.insert("0.0", "Predictions will appear here...\n")

    def start_prediction(self):
        url = self.url_entry.get()
        if not url:
            self.log("Please enter a URL.")
            return

        self.predict_button.configure(state="disabled")
        self.log(f"Starting prediction for: {url}")
        
        # Run in a separate thread to keep GUI responsive
        threading.Thread(target=self.run_process, args=(url,), daemon=True).start()

    def run_process(self, url):
        try:
            race_data = scraper.fetch_race_data(url)
            self.log("Data fetched successfully.")
            
            result = predictor.predict(race_data)
            self.log("Prediction complete.")
            self.log(f"\nResults:\n{result}")
            
        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.predict_button.configure(state="normal")

    def log(self, message):
        self.output_textbox.insert("end", message + "\n")
        self.output_textbox.see("end")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
    
    app = KeibaApp()
    app.mainloop()
