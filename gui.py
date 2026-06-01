import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox)
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtCore import Qt

# Data training
dataset = [
    {"No": 1, "Usia": "Muda", "Penghasilan": "Tinggi", "Status": "Single", "Membeli": "Ya"},
    {"No": 2, "Usia": "Tua", "Penghasilan": "Rendah", "Status": "Menikah", "Membeli": "Tidak"},
    {"No": 3, "Usia": "ParuhBaya", "Penghasilan": "Sedang", "Status": "Single", "Membeli": "Ya"},
    {"No": 4, "Usia": "Muda", "Penghasilan": "Rendah", "Status": "Single", "Membeli": "Tidak"},
    {"No": 5, "Usia": "Tua", "Penghasilan": "Sedang", "Status": "Menikah", "Membeli": "Tidak"},
    {"No": 6, "Usia": "ParuhBaya", "Penghasilan": "Tinggi", "Status": "Menikah", "Membeli": "Ya"},
    {"No": 7, "Usia": "Muda", "Penghasilan": "Sedang", "Status": "Single", "Membeli": "Ya"},
    {"No": 8, "Usia": "Tua", "Penghasilan": "Tinggi", "Status": "Menikah", "Membeli": "Tidak"},
    {"No": 9, "Usia": "ParuhBaya", "Penghasilan": "Rendah", "Status": "Single", "Membeli": "Ya"},
    {"No": 10, "Usia": "Muda", "Penghasilan": "Tinggi", "Status": "Menikah", "Membeli": "Ya"},
    {"No": 11, "Usia": "Tua", "Penghasilan": "Rendah", "Status": "Single", "Membeli": "Tidak"},
    {"No": 12, "Usia": "ParuhBaya", "Penghasilan": "Sedang", "Status": "Menikah", "Membeli": "Ya"},
    {"No": 13, "Usia": "Muda", "Penghasilan": "Sedang", "Status": "Single", "Membeli": "Ya"},
    {"No": 14, "Usia": "Tua", "Penghasilan": "Tinggi", "Status": "Menikah", "Membeli": "Tidak"},
    {"No": 15, "Usia": "ParuhBaya", "Penghasilan": "Rendah", "Status": "Single", "Membeli": "Ya"},
    {"No": 16, "Usia": "Muda", "Penghasilan": "Rendah", "Status": "Menikah", "Membeli": "Tidak"},
    {"No": 17, "Usia": "Tua", "Penghasilan": "Sedang", "Status": "Single", "Membeli": "Tidak"},
    {"No": 18, "Usia": "ParuhBaya", "Penghasilan": "Tinggi", "Status": "Menikah", "Membeli": "Ya"},
    {"No": 19, "Usia": "Muda", "Penghasilan": "Sedang", "Status": "Single", "Membeli": "Ya"},
    {"No": 20, "Usia": "Tua", "Penghasilan": "Rendah", "Status": "Menikah", "Membeli": "Tidak"},
]

def calculate_naive_bayes(data_uji):
    total_data = len(dataset)
    jumlah_ya = sum(1 for d in dataset if d["Membeli"] == "Ya")
    jumlah_tidak = sum(1 for d in dataset if d["Membeli"] == "Tidak")
    
    prior_ya = jumlah_ya / total_data
    prior_tidak = jumlah_tidak / total_data
    
    def count_likelihood(atribut, nilai, kelas):
        return sum(1 for d in dataset if d[atribut] == nilai and d["Membeli"] == kelas)
    
    like_usia_ya = count_likelihood("Usia", data_uji["Usia"], "Ya") / jumlah_ya
    like_penghasilan_ya = count_likelihood("Penghasilan", data_uji["Penghasilan"], "Ya") / jumlah_ya
    like_status_ya = count_likelihood("Status", data_uji["Status"], "Ya") / jumlah_ya
    
    like_usia_tidak = count_likelihood("Usia", data_uji["Usia"], "Tidak") / jumlah_tidak
    like_penghasilan_tidak = count_likelihood("Penghasilan", data_uji["Penghasilan"], "Tidak") / jumlah_tidak
    like_status_tidak = count_likelihood("Status", data_uji["Status"], "Tidak") / jumlah_tidak
    
    posterior_ya = prior_ya * like_usia_ya * like_penghasilan_ya * like_status_ya
    posterior_tidak = prior_tidak * like_usia_tidak * like_penghasilan_tidak * like_status_tidak
    
    prediksi = "Ya" if posterior_ya > posterior_tidak else "Tidak"
    
    return {
        "prior": {"ya": prior_ya, "tidak": prior_tidak, "jml_ya": jumlah_ya, "jml_tidak": jumlah_tidak, "total": total_data},
        "likelihood": {
            "ya": {
                "usia": like_usia_ya, "usia_count": count_likelihood("Usia", data_uji["Usia"], "Ya"),
                "penghasilan": like_penghasilan_ya, "penghasilan_count": count_likelihood("Penghasilan", data_uji["Penghasilan"], "Ya"),
                "status": like_status_ya, "status_count": count_likelihood("Status", data_uji["Status"], "Ya")
            },
            "tidak": {
                "usia": like_usia_tidak, "usia_count": count_likelihood("Usia", data_uji["Usia"], "Tidak"),
                "penghasilan": like_penghasilan_tidak, "penghasilan_count": count_likelihood("Penghasilan", data_uji["Penghasilan"], "Tidak"),
                "status": like_status_tidak, "status_count": count_likelihood("Status", data_uji["Status"], "Tidak")
            }
        },
        "posterior": {
            "ya": posterior_ya,
            "tidak": posterior_tidak
        },
        "prediksi": prediksi,
        "data_uji": data_uji
    }

class NaiveBayesApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Naive Bayes - Desktop Version")
        self.setGeometry(100, 100, 1100, 700)
        
        # Main Widget and Layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # --- LEFT PANEL: DATASET TABLE ---
        left_panel = QVBoxLayout()
        table_label = QLabel("Data Training (20 Baris)")
        table_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        self.table = QTableWidget()
        self.table.setRowCount(len(dataset))
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["No", "Usia", "Penghasilan", "Status", "Membeli"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        for row_idx, row_data in enumerate(dataset):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row_data["No"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(row_data["Usia"]))
            self.table.setItem(row_idx, 2, QTableWidgetItem(row_data["Penghasilan"]))
            self.table.setItem(row_idx, 3, QTableWidgetItem(row_data["Status"]))
            self.table.setItem(row_idx, 4, QTableWidgetItem(row_data["Membeli"]))
            
        left_panel.addWidget(table_label)
        left_panel.addWidget(self.table)
        
        # --- RIGHT PANEL: FORM & RESULTS ---
        right_panel = QVBoxLayout()
        
        # Form GroupBox
        form_group = QGroupBox("Form Uji Prediksi Baru")
        form_group.setFont(QFont("Arial", 11, QFont.Bold))
        form_layout = QVBoxLayout()
        
        # Usia Input
        h_usia = QHBoxLayout()
        h_usia.addWidget(QLabel("Usia:"))
        self.combo_usia = QComboBox()
        self.combo_usia.addItems(["Muda", "ParuhBaya", "Tua"])
        self.combo_usia.setCurrentText("ParuhBaya") # Default
        h_usia.addWidget(self.combo_usia)
        
        # Penghasilan Input
        h_penghasilan = QHBoxLayout()
        h_penghasilan.addWidget(QLabel("Penghasilan:"))
        self.combo_penghasilan = QComboBox()
        self.combo_penghasilan.addItems(["Rendah", "Sedang", "Tinggi"])
        self.combo_penghasilan.setCurrentText("Rendah") # Default
        h_penghasilan.addWidget(self.combo_penghasilan)
        
        # Status Input
        h_status = QHBoxLayout()
        h_status.addWidget(QLabel("Status:"))
        self.combo_status = QComboBox()
        self.combo_status.addItems(["Single", "Menikah"])
        self.combo_status.setCurrentText("Menikah") # Default
        h_status.addWidget(self.combo_status)
        
        # Submit Button
        self.btn_predict = QPushButton("Hitung Prediksi")
        self.btn_predict.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_predict.setStyleSheet("background-color: #6d28d9; color: white; padding: 10px; border-radius: 5px;")
        self.btn_predict.setCursor(Qt.PointingHandCursor)
        self.btn_predict.clicked.connect(self.run_prediction)
        
        form_layout.addLayout(h_usia)
        form_layout.addLayout(h_penghasilan)
        form_layout.addLayout(h_status)
        form_layout.addWidget(self.btn_predict)
        form_group.setLayout(form_layout)
        
        # Result Display Area
        result_label = QLabel("Rincian Perhitungan:")
        result_label.setFont(QFont("Arial", 11, QFont.Bold))
        
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 11))
        self.text_result.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px;")
        
        right_panel.addWidget(form_group)
        right_panel.addSpacing(20)
        right_panel.addWidget(result_label)
        right_panel.addWidget(self.text_result)
        
        # Combine Left and Right
        main_layout.addLayout(left_panel, 50) # 50% width
        main_layout.addLayout(right_panel, 50) # 50% width
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Run default prediction on startup
        self.run_prediction()

    def run_prediction(self):
        data_uji = {
            "Usia": self.combo_usia.currentText(),
            "Penghasilan": self.combo_penghasilan.currentText(),
            "Status": self.combo_status.currentText()
        }
        
        h = calculate_naive_bayes(data_uji)
        
        output = []
        output.append("=== 1. PROBABILITAS PRIOR ===")
        output.append(f"P(Membeli = Ya)    = {h['prior']['jml_ya']}/{h['prior']['total']} = {h['prior']['ya']:.2f}")
        output.append(f"P(Membeli = Tidak) = {h['prior']['jml_tidak']}/{h['prior']['total']} = {h['prior']['tidak']:.2f}\n")
        
        output.append("=== 2. LIKELIHOOD ===")
        output.append(f"[Kelas: Ya]")
        output.append(f"P(Usia={data_uji['Usia']} | Ya)        = {h['likelihood']['ya']['usia_count']}/{h['prior']['jml_ya']} = {h['likelihood']['ya']['usia']:.3f}")
        output.append(f"P(Penghasilan={data_uji['Penghasilan']} | Ya) = {h['likelihood']['ya']['penghasilan_count']}/{h['prior']['jml_ya']} = {h['likelihood']['ya']['penghasilan']:.3f}")
        output.append(f"P(Status={data_uji['Status']} | Ya)      = {h['likelihood']['ya']['status_count']}/{h['prior']['jml_ya']} = {h['likelihood']['ya']['status']:.3f}\n")
        
        output.append(f"[Kelas: Tidak]")
        output.append(f"P(Usia={data_uji['Usia']} | Tidak)        = {h['likelihood']['tidak']['usia_count']}/{h['prior']['jml_tidak']} = {h['likelihood']['tidak']['usia']:.3f}")
        output.append(f"P(Penghasilan={data_uji['Penghasilan']} | Tidak) = {h['likelihood']['tidak']['penghasilan_count']}/{h['prior']['jml_tidak']} = {h['likelihood']['tidak']['penghasilan']:.3f}")
        output.append(f"P(Status={data_uji['Status']} | Tidak)      = {h['likelihood']['tidak']['status_count']}/{h['prior']['jml_tidak']} = {h['likelihood']['tidak']['status']:.3f}\n")
        
        output.append("=== 3. POSTERIOR PROBABILITY ===")
        output.append(f"P(Ya | X)    ∝ {h['prior']['ya']:.2f} * {h['likelihood']['ya']['usia']:.3f} * {h['likelihood']['ya']['penghasilan']:.3f} * {h['likelihood']['ya']['status']:.3f} = {h['posterior']['ya']:.5f}")
        output.append(f"P(Tidak | X) ∝ {h['prior']['tidak']:.2f} * {h['likelihood']['tidak']['usia']:.3f} * {h['likelihood']['tidak']['penghasilan']:.3f} * {h['likelihood']['tidak']['status']:.3f} = {h['posterior']['tidak']:.5f}\n")
        
        output.append("=== 4. KESIMPULAN ===")
        output.append(f"Karena nilai posterior {h['prediksi']} lebih besar, maka pelanggan diprediksi:")
        output.append(f">>> {h['prediksi'].upper()} <<<")
        
        self.text_result.setText("\n".join(output))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set default styling globally
    app.setStyle("Fusion")
    
    window = NaiveBayesApp()
    window.show()
    sys.exit(app.exec_())
