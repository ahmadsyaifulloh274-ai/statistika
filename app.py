from flask import Flask, render_template

app = Flask(__name__)

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
    # 1. Menghitung Probabilitas Prior
    total_data = len(dataset)
    jumlah_ya = sum(1 for d in dataset if d["Membeli"] == "Ya")
    jumlah_tidak = sum(1 for d in dataset if d["Membeli"] == "Tidak")
    
    prior_ya = jumlah_ya / total_data
    prior_tidak = jumlah_tidak / total_data
    
    # 2. Menghitung Likelihood
    def count_likelihood(atribut, nilai, kelas):
        return sum(1 for d in dataset if d[atribut] == nilai and d["Membeli"] == kelas)
    
    # Likelihood Ya
    like_usia_ya = count_likelihood("Usia", data_uji["Usia"], "Ya") / jumlah_ya
    like_penghasilan_ya = count_likelihood("Penghasilan", data_uji["Penghasilan"], "Ya") / jumlah_ya
    like_status_ya = count_likelihood("Status", data_uji["Status"], "Ya") / jumlah_ya
    
    # Likelihood Tidak
    like_usia_tidak = count_likelihood("Usia", data_uji["Usia"], "Tidak") / jumlah_tidak
    like_penghasilan_tidak = count_likelihood("Penghasilan", data_uji["Penghasilan"], "Tidak") / jumlah_tidak
    like_status_tidak = count_likelihood("Status", data_uji["Status"], "Tidak") / jumlah_tidak
    
    # 3. Menghitung Posterior Probability
    posterior_ya = prior_ya * like_usia_ya * like_penghasilan_ya * like_status_ya
    posterior_tidak = prior_tidak * like_usia_tidak * like_penghasilan_tidak * like_status_tidak
    
    # Kesimpulan
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

@app.route('/')
def index():
    data_uji = {"Usia": "ParuhBaya", "Penghasilan": "Rendah", "Status": "Menikah"}
    hasil = calculate_naive_bayes(data_uji)
    return render_template('index.html', dataset=dataset, hasil=hasil)

if __name__ == '__main__':
    app.run(debug=True)
