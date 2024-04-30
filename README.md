# Paques True AI
<br />
<br />
FibrCorp
<br />
<br />

# Authors
- Daniel Satria
- Ruby Abdullah

## How to Setup
From source:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Docker:
```
docker compose build
```
## How to Run
From source:
```
streamlit run app.py
```
Docker:
```
docker compose up -d
```
## Configuration
you can check or do some config on our configuration
- config/config.yml (for software configuration)
- docker-compose.yml (for deployment configuration)

Enhancement True AI:
- save model kedalam folder
- save output prediction ke dalam database
- mengganti warna menjadi serupa dengan pds
- membuat menu dashboard supaya bisa memilih antara pds atau TrueAI
- mengintegrasikan data

To-do:
- integrasi data ke pds

Catatan DONE:
- tidak boleh ada istilah paques -> EYRE (tentatif bisa diganti) -> sementara no label
- istilah pds diganti menjadi data processor
- trueAI -> data modeler
- logo paques di white label
- fungsi sederhana untuk mengecek apakah datanya valid -> bisa delete is_null
- scaling and any encoder bisa disimpan didalam pipeline model
- output prediksi bisa disimpan ke database
- membuat yaml config untuk database management and easy deployment
- kenapa harus konek ke database -> bisa ditampilkan kedalam dashboard mereka
- membuat log prediksi dashboard
- output feature engineering disimpan ke database
- output data exploration disimpan ke database:
- modul integrasi data ke pds
- end to end testing and build try except

- build docker and docker-compose
- buat akun github pakai fibrcorp,
- setup ssh github
- push it to that to

password github:qtcayqur246dxkt

Catatan:
- diganti table
data_exploration_{uid}_{nama_datasets}_{raw_data/correlation/data_description}
data_exploration_master
uid, nama, datasets, time_stamp
    - prediction
    - exploration
    - feature engineering
- database dengan starrock
- pull from the server
- try to run and access it
- also access it using dbeaver for postgres
- try to access some data from pds