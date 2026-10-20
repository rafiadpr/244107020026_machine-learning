# Jalankan file ini di Pydroid3.
LINGKUNGAN = globals().get('LINGKUNGAN', 'Pydroid3 - Redmi Note 9 Pro')

import numpy as np
import time
import importlib
import json
import platform
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Dataset dan parameter sesuai modul
np.random.seed(42)
n_samples = 10000
d = 128
k = 10
X = np.random.random((n_samples, d)).astype('float32')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hasil = []
versi = {'numpy': np.__version__, 'sklearn': importlib.import_module('sklearn').__version__}

def catat(model, waktu=None, error=None):
    hasil.append({'Model': model, 'Waktu (detik)': waktu,
                  'Status': 'Berhasil' if error is None else str(error)})
    if error is None:
        print(f'{model} done in {waktu:.3f} s')
    else:
        print(f'{model}: tidak dijalankan ({error})')

def muat_library(nama):
    try:
        lib = importlib.import_module(nama)
        versi[nama] = getattr(lib, '__version__', 'tidak tersedia')
        return lib
    except ImportError as error:
        return str(error)

# Exact NN (brute-force)
start = time.perf_counter()
nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
nn.fit(X_scaled)
dist_exact, idx_exact = nn.kneighbors(X_scaled)
time_exact = time.perf_counter() - start
catat('Exact NN', time_exact)

# Annoy
annoy = muat_library('annoy')
if isinstance(annoy, str):
    catat('Annoy', error=annoy)
else:
    start = time.perf_counter()
    f = X_scaled.shape[1]
    index_annoy = annoy.AnnoyIndex(f, 'euclidean')
    for i, v in enumerate(X_scaled):
        index_annoy.add_item(i, v)
    index_annoy.build(10)
    idx_annoy = [index_annoy.get_nns_by_vector(v, k) for v in X_scaled]
    time_annoy = time.perf_counter() - start
    catat('Annoy', time_annoy)

# HNSW
hnswlib = muat_library('hnswlib')
if isinstance(hnswlib, str):
    catat('HNSW', error=hnswlib)
else:
    start = time.perf_counter()
    p_hnsw = hnswlib.Index(space='l2', dim=d)
    p_hnsw.init_index(max_elements=n_samples, ef_construction=200, M=16)
    p_hnsw.add_items(X_scaled)
    p_hnsw.set_ef(200)
    idx_hnsw, _ = p_hnsw.knn_query(X_scaled, k=k)
    time_hnsw = time.perf_counter() - start
    catat('HNSW', time_hnsw)

# FAISS IVF
faiss = muat_library('faiss')
if isinstance(faiss, str):
    catat('FAISS IVF', error=faiss)
else:
    start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    index_faiss = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
    index_faiss.train(X_scaled)
    index_faiss.add(X_scaled)
    index_faiss.nprobe = 10
    _, idx_faiss = index_faiss.search(X_scaled, k)
    time_faiss = time.perf_counter() - start
    catat('FAISS IVF', time_faiss)

print('\n=== Ringkasan Waktu (detik) ===')
print('Lingkungan:', LINGKUNGAN)
for baris in hasil:
    waktu = baris['Waktu (detik)']
    print(f"{baris['Model']:10}: {waktu:.3f}" if waktu is not None
          else f"{baris['Model']:10}: N/A - {baris['Status']}")

# Simpan hasil asli agar dapat dibandingkan di notebook.
nama_file = ('hasil_colab.json' if LINGKUNGAN == 'Google Colab' else
             'hasil_redmi_note_9_pro.json' if LINGKUNGAN.startswith('Pydroid3') else
             'hasil_uji_lokal.json')
laporan = {
    'lingkungan': LINGKUNGAN,
    'perangkat': 'Redmi Note 9 Pro; RAM 6 GB; 64 GB; Android 12; MIUI 14'
                 if LINGKUNGAN.startswith('Pydroid3') else platform.platform(),
    'python': platform.python_version(), 'cpu': platform.processor(),
    'versi_library': versi,
    'parameter': {'n_samples': n_samples, 'dimensi': d, 'k': k, 'seed': 42},
    'hasil': hasil
}
Path(nama_file).write_text(json.dumps(laporan, indent=2), encoding='utf-8')
print('Hasil tersimpan di:', Path(nama_file).resolve())
