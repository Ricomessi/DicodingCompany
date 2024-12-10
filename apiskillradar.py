from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
from flask_cors import CORS

# === Setup Flask App ===
app = Flask(__name__)
CORS(app)

# === Load Data ===
learning_path_data = pd.read_csv('data/learningpath.csv')
user_data = pd.read_csv('data/userdata.csv')

# Konversi graduated_course_ids ke list of integers
user_data['graduated_course_ids'] = user_data['graduated_course_ids'].apply(
    lambda x: list(map(int, x.strip('[]').split(','))) if pd.notnull(x) else []
)

# Membuat matriks pengguna-kursus
user_course_matrix = pd.DataFrame(0, index=user_data['user_name'], columns=learning_path_data['Course ID'].unique())
for i, row in user_data.iterrows():
    user_course_matrix.loc[row['user_name'], row['graduated_course_ids']] = 1

# === Memuat model KNN dari file ===
model_knn = joblib.load('model/model_knn.pkl')

def get_skill_radar_data(user_completed_courses):
    """
    Menghitung data Skill Radar untuk pengguna
    """
    learning_path_data['Completed'] = learning_path_data['Course ID'].apply(
        lambda x: 1 if x in user_completed_courses else 0
    )
    completions = learning_path_data.groupby('Learning Path')['Completed'].sum()
    totals = learning_path_data.groupby('Learning Path')['Course ID'].count()
    labels = completions.index.tolist()
    values = [c / t for c, t in zip(completions, totals)]
    
    return {
        "labels": labels,
        "values": values,
        "max_values": [1] * len(labels)
    }

def get_recommended_courses_from_radar(user_completed_courses, completions, totals):
    """
    Mendapatkan rekomendasi kursus berdasarkan Skill Radar
    """
    low_completion_paths = completions[completions < totals].index.tolist()
    recommended_courses_from_radar = set()
    for path in low_completion_paths:
        path_courses = learning_path_data[
            (learning_path_data['Learning Path'] == path) & 
            (~learning_path_data['Course ID'].isin(user_completed_courses))
        ]['Course ID']
        recommended_courses_from_radar.update(path_courses)

    return list(recommended_courses_from_radar)[:5]

def get_recommended_courses_from_knn(user_name, n_neighbors, user_completed_courses):
    """
    Mendapatkan rekomendasi kursus berdasarkan KNN
    """
    recommended_courses_from_knn = set()
    if user_name in user_course_matrix.index:
        user_index = user_course_matrix.index.get_loc(user_name)
        distances, indices = model_knn.kneighbors(
            user_course_matrix.iloc[user_index, :].values.reshape(1, -1), 
            n_neighbors=n_neighbors + 1
        )
        
        for i in range(1, len(distances.flatten())):
            similar_user = user_course_matrix.index[indices.flatten()[i]]
            similar_user_courses = user_data[user_data['user_name'] == similar_user]['graduated_course_ids'].values[0]
            recommended_courses_from_knn.update(similar_user_courses)

        # Filter hanya course yang belum diselesaikan
        recommended_courses_from_knn = recommended_courses_from_knn - user_completed_courses

    return list(recommended_courses_from_knn)[:5]

@app.route('/user-progress', methods=['POST'])
def user_progress():
    """
    Endpoint untuk mendapatkan data Skill Radar (JSON) dan rekomendasi kursus
    """
    data = request.json
    user_name = data.get("user_name")
    n_recommendations = 10  # Total 10 rekomendasi
    n_neighbors = data.get("n_neighbors", 5)

    if not user_name:
        return jsonify({"error": "User name is required"}), 400

    # === Skill Radar Data ===
    user_courses = user_data[user_data['user_name'] == user_name]
    if user_courses.empty:
        return jsonify({"error": f"Nama pengguna '{user_name}' tidak ditemukan."}), 404

    user_completed_courses = set(user_courses.iloc[0]['graduated_course_ids'])
    
    # Skill Radar Calculation
    radar_data = get_skill_radar_data(user_completed_courses)
    
    # === Course Recommendations ===
    completions = learning_path_data.groupby('Learning Path')['Completed'].sum()
    totals = learning_path_data.groupby('Learning Path')['Course ID'].count()

    # Prioritas 1: Penyeimbangan Skill Radar (5 rekomendasi pertama)
    recommended_courses_from_radar = get_recommended_courses_from_radar(
        user_completed_courses, completions, totals
    )

    # Prioritas 2: Rekomendasi KNN (5 rekomendasi berikutnya)
    recommended_courses_from_knn = get_recommended_courses_from_knn(
        user_name, n_neighbors, user_completed_courses
    )

    # Menggabungkan hasil rekomendasi
    all_recommended_courses = recommended_courses_from_radar + recommended_courses_from_knn
    all_recommended_courses = list(set(all_recommended_courses))

    # Tambahkan kursus populer jika rekomendasi kurang
    remaining_recommendations = n_recommendations - len(all_recommended_courses)
    if remaining_recommendations > 0:
        popular_courses = learning_path_data['Course ID'].value_counts().index.tolist()
        additional_courses = [c for c in popular_courses if c not in user_completed_courses]
        all_recommended_courses += additional_courses[:remaining_recommendations]

    # Ambil course ID final dan nama course
    recommended_course_ids = all_recommended_courses[:n_recommendations]
    recommended_course_names = learning_path_data[
        learning_path_data['Course ID'].isin(recommended_course_ids)
    ]['Course Name'].unique().tolist()

    # === Response JSON ===
    response = {
        "user_name": user_name,
        "radar_data": radar_data,
        "recommended_courses": recommended_course_names
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
