# Laporan Proyek Machine Learning - Jessindy Tanuwijaya

![Pisang](https://raw.githubusercontent.com/jessdytan/predictive-analysis-dicoding/main/images/banana_quality.jpeg)

## Domain Proyek

Buah pisang merupakan salah satu buah tropis yang paling banyak dikonsumsi di seluruh dunia. Kualitas buah pisang yang baik sangat penting, terutama dalam industri pertanian, distribusi, dan konsumsi. Dalam praktiknya, penilaian kualitas buah pisang masih sering dilakukan secara manual, yang cenderung subjektif, memakan waktu, dan tidak konsisten.
Masalah ini penting untuk diselesaikan karena kualitas buah berpengaruh terhadap daya jual dan kepuasan konsumen. Dengan mengembangkan sistem prediksi kualitas berbasis machine learning, pelaku industri dapat mengotomatisasi proses penilaian kualitas buah secara objektif dan efisien.

Dataset yang digunakan dalam proyek ini berisi data kuantitatif dari berbagai parameter buah pisang seperti berat, panjang, ketebalan, dan tingkat keasaman (pH), dll. Parameter-parameter ini dapat digunakan untuk mengembangkan model klasifikasi yang dapat membedakan kualitas buah pisang secara akurat dan konsisten.

Penelitian sebelumnya oleh Soltani dkk. (2011) mendemonstrasikan bagaimana sifat dielektrik pisang dapat digunakan untuk mengevaluasi tingkat kematangan buah pisang, yang merupakan faktor penting dalam penilaian kualitas [^1].
Selain itu, Mustafa dkk. (2008) telah mengembangkan sistem pengolahan citra untuk menentukan ukuran dan tingkat kematangan pisang, yang menunjukkan hasil yang menjanjikan untuk otomatisasi grading buah pisang [^2].
Dalam konteks klasifikasi buah berdasarkan tekstur dan fitur bentuk-ukuran, Muhammad (2015) telah menunjukkan keberhasilan pendekatan ini pada klasifikasi buah kurma, yang metodologinya dapat diadaptasi untuk klasifikasi kualitas pisang [^3].

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi kualitas buah pisang secara otomatis dan akurat?
2. Apa pengaruh dari fitur-fitur seperti kemanisan, kelembutan, dan waktu panen terhadap kualitas buah?
3. Apakah metode machine learning lebih efektif dibandingkan penilaian manual dalam menilai kualitas buah?
4. Bagaimana cara membandingkan performa berbagai algoritma klasifikasi untuk memprediksi kualitas?

### Goals

1. Mengembangkan model prediktif berbasis machine learning untuk mengklasifikasikan kualitas buah pisang.
2. Mengukur kontribusi fitur-fitur terhadap kualitas buah pisang.
3. Membuktikan bahwa pendekatan berbasis machine learning memberikan hasil yang lebih konsisten daripada metode manual [^2].
4. Menganalisis dan membandingkan performa beberapa algoritma klasifikasi (Logistic Regression, KNN, Random Forest).
5. Meningkatkan efisiensi proses penilaian buah di sektor pertanian.

### Solution Statements

- Mengembangkan tiga model machine learning: Logistic Regression, K-Nearest Neighbors, dan Random Forest Classifier untuk klasifikasi kualitas buah.
- Melakukan evaluasi performa dengan metrik akurasi, precision, recall, dan F1-score [^4].
- Memilih model terbaik berdasarkan hasil evaluasi dan analisis error.
- Melakukan normalisasi data dan pembagian dataset untuk validasi yang adil.
- Jika perlu, dilakukan hyperparameter tuning pada model Random Forest dan KNN untuk peningkatan performa.

## Data Understanding

| Jenis Informasi | Keterangan                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| Format          | .csv                                                                                                    |
| **Title**       | Banana Quality                                                                                          |
| **Source**      | Kaggle                                                                                                  |
| **Maintainer**  | l3LlFF                                                                                                  |
| **License**     | Apache 2.0                                                                                              |
| **Visibility**  | Publik                                                                                                  |
| **Tags**        | Earth and Nature, Education, Food, Data Visualization, Exploratory Data Analysis, Binary Classification |
| **Usability**   | 10.0                                                                                                    |

Dataset yang digunakan adalah “Dataset yang digunakan adalah [Banana Quality](https://www.kaggle.com/datasets/l3llff/banana)” dari Kaggle terdiri dari **8000 baris** data dan **8 kolom**. Dataset ini digunakan untuk menganalisis dan mengklasifikasikan kualitas buah pisang berdasarkan berbagai fitur fisik dan kimia seperti ukuran, berat, tingkat kemanisan, keasaman, dan tingkat kematangan. Tujuan utama dari dataset ini adalah membangun model klasifikasi biner yang dapat memprediksi apakah suatu buah pisang tergolong baik atau buruk berdasarkan karakteristik yang dimilikinya.

### Fitur-fitur pada dataset:

| Fitur         | Deskripsi                                 |
| ------------- | ----------------------------------------- |
| `Size`        | Ukuran buah                               |
| `Weight`      | Berat buah                                |
| `Sweetness`   | Tingkat kemanisan                         |
| `Softness`    | Tingkat kelembutan                        |
| `HarvestTime` | Waktu sejak panen (dalam satuan tertentu) |
| `Ripeness`    | Tingkat kematangan buah                   |
| `Acidity`     | Tingkat keasaman buah                     |
| `Quality`     | Kategori kualitas buah (target variabel)  |

### Contoh Data (Head)

| Size      | Weight   | Sweetness | Softness  | HarvestTime | Ripeness | Acidity  | Quality |
| --------- | -------- | --------- | --------- | ----------- | -------- | -------- | ------- |
| -1.924968 | 0.468078 | 3.077832  | -1.472177 | 0.294799    | 2.435570 | 0.271290 | Good    |
| -2.409751 | 0.486870 | 0.346921  | -2.495099 | -0.892213   | 2.067549 | 0.307325 | Good    |
| -0.357607 | 1.483176 | 1.568452  | -2.645145 | -0.647267   | 3.090643 | 1.427322 | Good    |
| -0.868524 | 1.566201 | 1.889605  | -1.273761 | -1.006278   | 1.873001 | 0.477862 | Good    |
| 0.651825  | 1.319199 | -0.022459 | -1.209709 | -1.430692   | 1.078345 | 2.812442 | Good    |

### Kondisi Data:

- Duplikat: **Tidak ditemukan baris duplikat dalam dataset.**
- Nilai Kosong: Dataset **tidak mengandung nilai kosong** pada kolom manapun.
- Tipe Data: Semua kolom telah memiliki tipe data yang sesuai.
  | No | Kolom | Tipe Data |
  |----|--------------|-----------|
  | 1 | Size | float64 |
  | 2 | Weight | float64 |
  | 3 | Sweetness | float64 |
  | 4 | Softness | float64 |
  | 5 | HarvestTime | float64 |
  | 6 | Ripeness | float64 |
  | 7 | Acidity | float64 |
  | 8 | Quality | object |
- Outlier: Terdapat data outlier pada kolom `Size`, `Weight`, `Sweetness`, `Softness`, `HarvestTime`, `Ripeness`, dan `Acidity` diidentifikasi menggunakan visualisasi boxplot dan ditangani dengan **metode IQR**. Berikut adalah visualisasi boxplot nya:
  ![Outlier](https://raw.githubusercontent.com/jessdytan/predictive-analysis-dicoding/main/images/outlier.png)
  Berdasarkan hasil visualisasi:
  - Semua fitur memiliki outlier, terlihat dari titik-titik di luar whisker.
  - Outlier paling banyak muncul di fitur HarvestTime, Ripeness, dan Acidity.
  - Sebaran data cukup simetris untuk sebagian besar fitur.
  - Boxplot menunjukkan bahwa meskipun ada outlier, rentang antar kuartil (IQR) masih wajar dan data tidak terlalu skew.

### Exploratory Data Analysis (EDA)

- Visualisasi distribusi setiap fitur menggunakan histogram.
  ![Distribusi Label](https://raw.githubusercontent.com/jessdytan/predictive-analysis-dicoding/main/images/hist.png)

  Berdasarkan visualisasi distribusi fitur dapat disimpulkan:

  - Size: Distribusi mendekati normal dengan sedikit skew ke kanan. Sebagian besar nilai berada di sekitar -2 hingga 1.
  - Weight: Distribusi cukup simetris dan menyerupai distribusi normal. Nilai terpusat di sekitar -1 hingga 1.
  - Sweetness: Terlihat seperti distribusi normal, meskipun sedikit skew ke kiri. Sebagian besar nilai berada antara -3 dan 1.
  - Softness: Distribusi bersifat bimodal (dua puncak), menunjukkan adanya dua kelompok dominan dalam data softness.
  - HarvestTime: Distribusi agak simetris dengan sedikit skew ke kanan. Sebagian besar nilai antara -3 dan 2.
  - Ripeness: Distribusi mendekati normal dengan sedikit skew ke kanan. Sebagian besar nilai berada antara -1 dan 3.
  - Acidity: Distribusi simetris menyerupai distribusi normal. Sebagian besar nilai berkisar dari -3 hingga 2.

  Secara keseluruhan, sebagian besar fitur memiliki distribusi mendekati normal, kecuali `Softness` yang menunjukkan pola bimodal yang menonjol dan bisa menunjukkan adanya segmentasi dalam data tersebut.

- Korelasi antar fitur menggunakan heatmap.
  ![Heatmap](https://raw.githubusercontent.com/jessdytan/predictive-analysis-dicoding/main/images/heatmap.png)

  Berdasarkan hasil visualisasi:

  - Size dan HarvestTime punya korelasi positif (0.58) → Pisang yang lebih besar cenderung dipanen lebih lama.
  - Weight dan Sweetness (0.40) serta Weight dan Acidity (0.43) → Pisang yang lebih berat cenderung lebih manis dan asam.
  - Ripeness dan Acidity berkorelasi negatif (-0.35) → Pisang makin matang, makin rendah keasamannya.
  - Sebagian besar fitur lainnya tidak berkorelasi kuat.
  - Tidak ada indikasi multikolinearitas, jadi semua fitur masih aman digunakan dalam model.

- Distribusi label `Quality` untuk memastikan tidak ada ketidakseimbangan kelas.
  ![Distribusi Label](https://raw.githubusercontent.com/jessdytan/predictive-analysis-dicoding/main/images/quality.png)

  Berdasarkan hasil visualisasi:

  - Data label cukup seimbang antara Good dan Bad.
  - Jumlah pisang Bad sedikit lebih banyak dibanding Good.
  - Distribusi seimbang seperti ini ideal untuk model klasifikasi, karena tidak berat sebelah.

## Data Preparation

Pada tahap ini, dilakukan beberapa teknik untuk menyiapkan data sebelum masuk ke proses pemodelan. Urutan dan penjelasan tiap langkah sebagai berikut:

1. Mengecek Data Duplikat
   Dilakukan pengecekan data duplikat untuk memastikan tidak ada pengulangan entri yang dapat memengaruhi akurasi model.
   Setelah dilakukan pengecekan, ternyata tidak ada data duplikat dalam dataset.
   **Alasan**: Duplikat dapat menyebabkan bias dalam model dan menurunkan performa karena informasi yang sama dihitung lebih dari sekali.

2. Mengecek Nilai yang Hilang (Missing Values)
   Dilakukan pengecekan nilai kosong, setelah dilakukan ternyata tidak ada data yang hilang.
   **Alasan**: Nilai kosong bisa mengganggu proses training model dan menurunkan kualitas prediksi.

3. Menangani Outlier
   Outlier dideteksi menggunakan visualisasi boxplot dan kemudian ditangani dengan metode IQR.
   **Alasan**: Outlier dapat memengaruhi distribusi data dan membuat model belajar pola yang tidak umum (noise).

4. Encoding Target (Quality)
   Label Quality awalnya berupa kategorikal ("Good", "Bad"), kemudian diubah menjadi numerik (misalnya 1 dan 0) menggunakan label encoding.
   **Alasan**: Model machine learning hanya dapat bekerja dengan data numerik untuk target prediksi.

5. Normalisasi Data (StandardScaler)
   Semua fitur numerik dinormalisasi menggunakan StandardScaler agar memiliki skala yang seragam.
   **Alasan**: Beberapa algoritma seperti KNN dan Logistic Regression sensitif terhadap skala data. Normalisasi membantu model mempelajari pola secara lebih optimal.

6. Drop Kolom `Softness`
   Berdasarkan analisis eksplorasi data (EDA) sebelumnya, telah dilakukan visualisasi distribusi data untuk memahami karakteristik variabel yang relevan. Hasil visualisasi menunjukkan bahwa kolom `Softness` memiliki distribusi yang bimodal, yang mengindikasikan adanya dua kelompok atau mode yang berbeda dalam data ini.
   **Alasan**: Dengan menghapus kolom ini, dapat menyederhanakan analisis dan memastikan fokus tetap pada aspek-aspek yang lebih relevan dan konsisten dengan tujuan analisis yang sedang dilakukan.

7. Split Data: Training dan Test Set (70:30)
   Dataset dibagi menjadi 70% data latih dan 30% data uji.

   **Alasan**: Pembagian ini penting untuk mengevaluasi performa model secara objektif terhadap data yang belum pernah dilihat sebelumnya.

## Modeling

Model yang digunakan:

1. **Logistic Regression**

   - **Cara kerja:** Logistic Regression bekerja dengan memodelkan probabilitas suatu data masuk ke dalam kelas tertentu menggunakan fungsi sigmoid. Fungsi ini mengubah nilai prediksi menjadi nilai antara 0 dan 1, yang kemudian digunakan untuk klasifikasi. Model ini mengasumsikan adanya hubungan linear antara variabel input dan logit dari probabilitas.
   - **Parameter:**
     - `max_iter=10000`: Jumlah maksimum iterasi untuk konvergensi. Default-nya adalah `100`, namun dinaikkan agar memastikan konvergensi pada dataset yang kompleks.
     - `solver='saga'`: Digunakan untuk optimasi, cocok untuk dataset besar dan mendukung regularisasi L1 dan L2.
   - **Tuning:** Tidak dilakukan tuning karena model ini digunakan sebagai baseline.
   - **Kelebihan:** Sederhana, cepat, dan hasilnya mudah diinterpretasi.
   - **Kekurangan:** Kurang efektif jika hubungan antar fitur dan target tidak bersifat linear.

2. **K-Nearest Neighbors (KNN)**

   - **Cara kerja:** KNN mengklasifikasikan data baru berdasarkan mayoritas label dari sejumlah tetangga terdekat (k). Jarak antara data dihitung (misalnya dengan Euclidean), kemudian label dari `k` data terdekat digunakan untuk menentukan kelas.
   - **Parameter default:** `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'`, `p=2`
   - **Hyperparameter tuning:**
     - Dilakukan menggunakan `GridSearchCV` dengan ruang pencarian:
       ```python
       param_grid = {
           'n_neighbors': [3, 5, 7, 9, 11, 13],
           'weights': ['uniform', 'distance'],
           'metric': ['euclidean', 'manhattan'],
           'p': [1, 2]
       }
       ```
     - **Hasil terbaik:** `{'metric': 'euclidean', 'n_neighbors': 13, 'p': 1, 'weights': 'distance'}`
   - **Kelebihan:** Non-parametrik, efektif untuk dataset kecil dan terstruktur.[^4]
   - **Kekurangan:** Boros memori dan lambat pada dataset besar, serta sensitif terhadap fitur yang tidak relevan.

3. **Random Forest Classifier**
   - **Cara kerja:** Random Forest adalah kumpulan dari banyak decision tree yang dilatih secara acak terhadap subset data dan fitur. Prediksi akhir diperoleh dengan voting mayoritas dari semua pohon. Metode ini mengurangi overfitting dan meningkatkan generalisasi model.
   - **Parameter:**
     - `n_estimators=100`: Jumlah pohon dalam forest. Default adalah `100`.
     - `max_depth=None`: Kedalaman maksimum pohon. `None` berarti pohon tumbuh sampai semua daun murni (default).
   - **Hyperparameter tuning:**
     - Dilakukan dengan `GridSearchCV` dengan ruang pencarian:
       ```python
       param_grid = {
           'n_estimators': [100, 200],
           'max_depth': [None, 10, 20],
           'min_samples_split': [2, 5],
           'min_samples_leaf': [1, 2],
           'bootstrap': [True, False]
       }
       ```
     - **Hasil terbaik:** `{'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}`
   - **Kelebihan:** Akurasi tinggi, mampu menangani data non-linear, dan menyediakan fitur penting (feature importance).[^5]
   - **Kekurangan:** Kurang interpretatif dibandingkan model sederhana, serta memiliki biaya komputasi yang tinggi.

### Pemilihan Model Terbaik

- Semua model diuji pada test set dan dibandingkan berdasarkan metrik klasifikasi.
- Random Forest dipilih sebagai model terbaik karena menghasilkan F1-score tertinggi dan memiliki toleransi terhadap data yang kompleks dan non-linear.

## Evaluation

### Metrik yang Digunakan:

- **Akurasi**

  $\displaystyle \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

  Akurasi mengukur proporsi prediksi yang benar terhadap keseluruhan data. Ini adalah metrik yang baik jika distribusi kelas seimbang. Karena dataset kualitas pisang ini relatif seimbang antara kelas 0 (tidak berkualitas) dan 1 (berkualitas), maka akurasi tetap relevan untuk digunakan.

- **Precision**

  $\displaystyle \text{Precision} = \frac{TP}{TP + FP}$

  Precision digunakan untuk mengukur seberapa banyak prediksi positif yang benar-benar positif. Ini penting jika kita ingin meminimalkan kesalahan dalam menyatakan buah sebagai "berkualitas" padahal tidak.

- **Recall**

  $\displaystyle \text{Recall} = \frac{TP}{TP + FN}$

  Recall mengukur seberapa banyak dari data aktual positif yang berhasil kita tangkap. Ini penting jika tujuan kita adalah tidak melewatkan buah yang seharusnya diklasifikasikan sebagai "berkualitas".

- **F1-Score**

  $\displaystyle F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

  F1-score adalah harmonic mean dari precision dan recall. Metrik ini berguna untuk memberikan keseimbangan antara precision dan recall, terutama saat keduanya sama-sama penting.

### Hasil Evaluasi

| Model                    | Accuracy | Precision | Recall | F1-Score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Logistic Regression      | 0.8755   | 0.875     | 0.875  | 0.8755   |
| KNN Classifier           | 0.9487   | 0.949     | 0.949  | 0.9487   |
| Random Forest Classifier | 0.9443   | 0.944     | 0.944  | 0.9443   |

### Hasil Evaluasi Setelah Tuning (GridSearchCV)

| Model                            | Accuracy   | Precision | Recall | F1-Score   |
| -------------------------------- | ---------- | --------- | ------ | ---------- |
| KNN Classifier (Tuned)           | **0.9530** | 0.9530    | 0.9530 | **0.9530** |
| Random Forest Classifier (Tuned) | 0.9439     | 0.9439    | 0.9439 | 0.9439     |

### Analisis Hasil

- **Model terbaik** berdasarkan hasil evaluasi adalah **KNN Classifier setelah tuning**, dengan akurasi dan F1-score tertinggi yaitu **0.9530**.
- **Logistic Regression** menunjukkan performa paling rendah, namun tetap berguna sebagai baseline yang cepat dan sederhana.
- **Random Forest** sangat kompetitif dengan KNN, namun tidak mengalami peningkatan signifikan setelah tuning.

### Evaluasi Terhadap Business Understanding

1. Tujuan utama adalah membangun model machine learning untuk memprediksi kualitas buah pisang dengan akurat dan efisien. Dari hasil evaluasi, model KNN setelah tuning menunjukkan performa terbaik dengan akurasi dan F1-score mencapai 95.3%.
2. - **Logistic Regression** menunjukkan performa paling rendah, namun tetap berguna sebagai baseline yang cepat dan sederhana.
   - **Random Forest** sangat kompetitif dengan KNN, namun tidak mengalami peningkatan signifikan setelah tuning.
3. Hal ini mendukung tujuan bisnis untuk mengurangi ketergantungan pada proses manual, karena model ini dapat memprediksi kualitas buah secara otomatis dengan hasil yang sangat mendekati penilaian manusia.
4. Evaluasi juga membantu menjawab pertanyaan:“Apakah machine learning lebih efektif dibandingkan metode manual?”
   Jawabannya adalah ya, karena model memberikan prediksi yang konsisten dan bisa direproduksi tanpa kelelahan atau subjektivitas seperti dalam evaluasi manual.
5. Precision dan recall yang tinggi menunjukkan bahwa fitur-fitur seperti kemanisan, kelembutan, dan waktu panen memang bisa diandalkan untuk menentukan kualitas buah. Ini menjawab pertanyaan tentang pengaruh fitur terhadap kualitas.
6. Dengan mengetahui bahwa model KNN unggul, maka bisa diambil keputusan untuk mengintegrasikan model ini ke dalam sistem sortir digital di sektor pertanian atau rantai pasok buah pisang.

## Kesimpulan

Proyek ini berhasil menunjukkan bahwa pendekatan machine learning berbasis fitur-fitur kuantitatif seperti ukuran, berat, kemanisan, keasaman, dan kelembutan dapat digunakan secara efektif untuk memprediksi kualitas buah pisang. Model K-Nearest Neighbors yang telah dituning terbukti menghasilkan performa terbaik dengan F1-score sebesar 0.9530, mengungguli Logistic Regression dan Random Forest. Hal ini membuktikan bahwa machine learning dapat memberikan solusi objektif dan efisien dalam proses klasifikasi kualitas buah, mengurangi ketergantungan terhadap metode manual yang bersifat subjektif. Ke depannya, model ini dapat diintegrasikan ke dalam sistem otomasi industri pertanian dan distribusi buah.

---

**Referensi:**
[^1]: Soltani, M., Alimardani, R., & Omid, M. (2011). "Evaluating banana ripening status from measuring dielectric properties." Journal of Food Engineering, 105(4), 625-631. https://doi.org/10.1016/j.jfoodeng.2011.03.032
[^2]: Mustafa, N. B. A., Fuad, N. A., Ahmed, S. K., Abidin, A. A. Z., Ali, Z., Yit, W. B., & Sharrif, Z. A. M. (2008). "Image processing of an agriculture produce: Determination of size and ripeness of a banana." 2008 International Symposium on Information Technology, 1-7. https://doi.org/10.1109/ITSIM.2008.4631636
[^3]: Muhammad, G. (2015). "Date fruits classification using texture descriptors and shape-size features." Engineering Applications of Artificial Intelligence, 37, 361-367. https://doi.org/10.1016/j.engappai.2014.10.001
[^4]: J. Brownlee, _Machine Learning Mastery With Python_, 2020.  
[^5]: L. Breiman, "Random Forests," _Machine Learning_, vol. 45, 2001.
