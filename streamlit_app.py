import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analisis Naive Bayes", page_icon="📊", layout="wide")

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

st.title("Aplikasi Naive Bayes - Web Version")
st.write("Prediksi Pembelian Produk Berdasarkan Usia, Penghasilan, dan Status")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data Training (20 Baris)")
    df = pd.DataFrame(dataset)
    st.dataframe(df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Form Uji Prediksi Baru")
    
    with st.form("prediction_form"):
        col_form1, col_form2, col_form3 = st.columns(3)
        with col_form1:
            usia_input = st.selectbox("Usia", ["Muda", "ParuhBaya", "Tua"], index=1)
        with col_form2:
            penghasilan_input = st.selectbox("Penghasilan", ["Rendah", "Sedang", "Tinggi"])
        with col_form3:
            status_input = st.selectbox("Status", ["Single", "Menikah"], index=1)
            
        submitted = st.form_submit_button("Hitung Prediksi", type="primary", use_container_width=True)
        
    st.subheader("Rincian Perhitungan")
    
    # Render calculation block initially with default or submitted values
    data_uji_dict = {
        "Usia": usia_input,
        "Penghasilan": penghasilan_input,
        "Status": status_input
    }
    h = calculate_naive_bayes(data_uji_dict)
    
    output = f"""=== 1. PROBABILITAS PRIOR ===
P(Membeli = Ya)    = {h['prior']['jml_ya']}/{h['prior']['total']} = {h['prior']['ya']:.2f}
P(Membeli = Tidak) = {h['prior']['jml_tidak']}/{h['prior']['total']} = {h['prior']['tidak']:.2f}

=== 2. LIKELIHOOD ===
[Kelas: Ya]
P(Usia={data_uji_dict['Usia']} | Ya)        = {h['likelihood']['ya']['usia_count']}/{h['prior']['jml_ya']} = {h['likelihood']['ya']['usia']:.3f}
P(Penghasilan={data_uji_dict['Penghasilan']} | Ya) = {h['likelihood']['ya']['penghasilan_count']}/{h['prior']['jml_ya']} = {h['likelihood']['ya']['penghasilan']:.3f}
P(Status={data_uji_dict['Status']} | Ya)      = {h['likelihood']['ya']['status_count']}/{h['prior']['jml_ya']} = {h['likelihood']['ya']['status']:.3f}

[Kelas: Tidak]
P(Usia={data_uji_dict['Usia']} | Tidak)        = {h['likelihood']['tidak']['usia_count']}/{h['prior']['jml_tidak']} = {h['likelihood']['tidak']['usia']:.3f}
P(Penghasilan={data_uji_dict['Penghasilan']} | Tidak) = {h['likelihood']['tidak']['penghasilan_count']}/{h['prior']['jml_tidak']} = {h['likelihood']['tidak']['penghasilan']:.3f}
P(Status={data_uji_dict['Status']} | Tidak)      = {h['likelihood']['tidak']['status_count']}/{h['prior']['jml_tidak']} = {h['likelihood']['tidak']['status']:.3f}

=== 3. POSTERIOR PROBABILITY ===
P(Ya | X)    ∝ {h['prior']['ya']:.2f} * {h['likelihood']['ya']['usia']:.3f} * {h['likelihood']['ya']['penghasilan']:.3f} * {h['likelihood']['ya']['status']:.3f} = {h['posterior']['ya']:.5f}
P(Tidak | X) ∝ {h['prior']['tidak']:.2f} * {h['likelihood']['tidak']['usia']:.3f} * {h['likelihood']['tidak']['penghasilan']:.3f} * {h['likelihood']['tidak']['status']:.3f} = {h['posterior']['tidak']:.5f}

=== 4. KESIMPULAN ===
Karena nilai posterior {h['prediksi']} lebih besar, maka pelanggan diprediksi:
>>> {h['prediksi'].upper()} <<<"""

    st.code(output, language="text")
