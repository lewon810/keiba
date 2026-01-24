import customtkinter as ctk
try:
    from . import scraper
    from . import predictor
except ImportError:
    import scraper
    import predictor
import threading
import datetime

class KeibaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Keiba Prediction App")
        self.geometry("900x700")

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(self, text="Keiba Prediction", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Tab View
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.tab_search = self.tab_view.add("Search Mode")
        self.tab_url = self.tab_view.add("Direct URL")
        
        # --- Search Tab ---
        self.setup_search_tab()

        # --- URL Tab ---
        self.url_label = ctk.CTkLabel(self.tab_url, text="Netkeiba URL:")
        self.url_label.pack(side="left", padx=10)
        
        self.url_entry = ctk.CTkEntry(self.tab_url, width=400, placeholder_text="Paste netkeiba shutuba table URL here")
        self.url_entry.pack(side="left", padx=10, fill="x", expand=True)

        self.predict_url_btn = ctk.CTkButton(self.tab_url, text="Predict", command=self.on_url_predict)
        self.predict_url_btn.pack(side="right", padx=10)

        # Output Area
        self.output_textbox = ctk.CTkTextbox(self, width=760, height=400)
        self.output_textbox.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.output_textbox.insert("0.0", "Predictions will appear here...\n")

    def setup_search_tab(self):
        # Date Input
        today = datetime.datetime.now().strftime("%Y%m%d")
        
        frame = ctk.CTkFrame(self.tab_search, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(frame, text="Date (YYYYMMDD):").pack(side="left", padx=5)
        self.date_entry = ctk.CTkEntry(frame, width=100)
        self.date_entry.insert(0, today)
        self.date_entry.pack(side="left", padx=5)
        
        # Place Dropdown
        # JRA Places: 01-10
        places = ["All", "01:Sapporo", "02:Hakodate", "03:Fukushima", "04:Niigata", "05:Tokyo", "06:Nakayama", "07:Chukyo", "08:Kyoto", "09:Hanshin", "10:Kokura"]
        ctk.CTkLabel(frame, text="Place:").pack(side="left", padx=5)
        self.place_option = ctk.CTkOptionMenu(frame, values=places, width=120)
        self.place_option.set("All")
        self.place_option.pack(side="left", padx=5)
        
        # Race Dropdown
        races = ["All"] + [str(i) for i in range(1, 13)]
        ctk.CTkLabel(frame, text="Race No:").pack(side="left", padx=5)
        self.race_option = ctk.CTkOptionMenu(frame, values=races, width=80)
        self.race_option.set("All")
        self.race_option.pack(side="left", padx=5)
        
        self.search_predict_btn = ctk.CTkButton(frame, text="Search & Predict", command=self.on_search_predict)
        self.search_predict_btn.pack(side="right", padx=10)

    def on_url_predict(self):
        url = self.url_entry.get()
        if not url:
            self.log("Please enter a URL.")
            return
        self.start_thread([url])

    def on_search_predict(self):
        date_str = self.date_entry.get()
        place_val = self.place_option.get()
        race_val = self.race_option.get()
        
        p_code = None
        if place_val != "All":
            p_code = place_val.split(":")[0]
            
        r_no = None
        if race_val != "All":
            r_no = race_val
            
        self.log(f"Searching races for {date_str} (Place: {place_val}, Race: {race_val})...")
        self.search_predict_btn.configure(state="disabled")
        
        threading.Thread(target=self.run_search_and_predict, args=(date_str, p_code, r_no), daemon=True).start()

    def run_search_and_predict(self, date_str, p_code, r_no):
        try:
            races = scraper.search_races(date_str, p_code, r_no)
            if not races:
                self.log("No races found matching the criteria.")
                return
                
            self.log(f"Found {len(races)} races. Starting prediction...")
            
            for r in races:
                self.log(f"\n--- {r['title']} ---")
                self.run_process_logic(r['url'])
                
        except Exception as e:
             self.log(f"Search Error: {e}")
        finally:
             self.search_predict_btn.configure(state="normal")

    def start_thread(self, urls):
        self.predict_url_btn.configure(state="disabled")
        threading.Thread(target=self.run_process_list, args=(urls,), daemon=True).start()

    def run_process_list(self, urls):
        try:
            for url in urls:
                self.log(f"Predicting: {url}")
                self.run_process_logic(url)
        finally:
            self.predict_url_btn.configure(state="normal")

    def run_process_logic(self, url):
        try:
            race_data = scraper.fetch_race_data(url)
            if not race_data:
                self.log("Failed to fetch data or empty.")
                return 
            
            result = predictor.predict(race_data)
            self.log(f"{result}")
            
        except Exception as e:
            self.log(f"Error: {e}")

    def log(self, message):
        self.output_textbox.insert("end", message + "\n")
        self.output_textbox.see("end")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = KeibaApp()
    app.mainloop()
