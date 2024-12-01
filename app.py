from flask import Flask, jsonify, request
from flask_cors import CORS
from supabase import create_client
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Kết nối đến Supabase
SUPABASE_URL = "https://kburxjzaitqoesdzkkiq.supabase.co"  # Thay bằng URL dự án của bạn
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtidXJ4anphaXRxb2VzZHpra2lxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxOTc1ODExMCwiZXhwIjoyMDM1MzM0MTEwfQ.Ek0dBEgg9sJshaHUSWaW00_C_mPwVuwdWDTPtTg5qfM"  # Thay bằng API key của bạn
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)

# Hàm lấy dữ liệu từ Supabase
def fetch_data():
    """
    Lấy dữ liệu từ bảng `movieViewingHistory` và `listMovieFavorite` trong Supabase.
    """
    # Lấy lịch sử xem phim (thêm cột `name`)
    watch_history = supabase.table("movieViewingHistory").select("userId, slug, name").execute()
    history_df = pd.DataFrame(watch_history.data)

    # Lấy danh sách phim yêu thích (thêm cột `name`)
    favorite_movies = supabase.table("listMovieFavorite").select("userId, slug, name").execute()
    favorite_df = pd.DataFrame(favorite_movies.data)

    return history_df, favorite_df

# Hàm gợi ý phim sử dụng thuật toán KNN
def recommend_movies(user_id, history_df, favorite_df, k=1, limit=10):
    """
    Gợi ý phim cho người dùng dựa trên lịch sử xem phim và phim yêu thích.
    Args:
        user_id (int): ID của người dùng cần gợi ý.
        history_df (DataFrame): Dữ liệu lịch sử xem phim.
        favorite_df (DataFrame): Dữ liệu phim yêu thích.
        k (int): Số người dùng gần nhất để so sánh.
        limit (int): Số lượng kết quả gợi ý tối đa.
    Returns:
        list: Danh sách các đối tượng {slug: ..., name: ...} của phim được gợi ý.
    """
    user_movie_matrix = history_df.pivot_table(index="userId", columns="slug", aggfunc="size", fill_value=0)

    model = NearestNeighbors()
    model.fit(user_movie_matrix)

    distances, indices = model.kneighbors([user_movie_matrix.loc[user_id]], n_neighbors=k + 1)
    similar_users = user_movie_matrix.index[indices.flatten()].tolist()

    similar_users.remove(user_id)

    recommended_movies = set()
    for similar_user in similar_users:
        # Gợi ý từ danh sách yêu thích
        user_favorites = favorite_df[favorite_df["userId"] == similar_user][["slug", "name"]].to_records(index=False)
        recommended_movies.update((row.slug, row.name) for row in user_favorites)

        # Gợi ý từ lịch sử xem
        user_history = history_df[history_df["userId"] == similar_user][["slug", "name"]].to_records(index=False)
        recommended_movies.update((row.slug, row.name) for row in user_history)

    # Loại bỏ các phim đã xem hoặc yêu thích bởi người dùng hiện tại
    watched_movies = history_df[history_df["userId"] == user_id]["slug"].tolist()
    favorite_movies = favorite_df[favorite_df["userId"] == user_id]["slug"].tolist()

    excluded_movies = set(watched_movies + favorite_movies)
    final_recommendations = [
        {"slug": slug, "name": name}
        for slug, name in recommended_movies
        if slug not in excluded_movies
    ]

    # Giới hạn số lượng kết quả trả về
    return final_recommendations[:limit]

# API gợi ý phim
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API nhận `user_id` qua query string và trả về danh sách các đối tượng {slug: ..., name: ...} của phim được gợi ý.
    """
    try:
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return jsonify({"status": "error", "message": "Missing user_id parameter"}), 400

        history_df, favorite_df = fetch_data()
        recommendations = recommend_movies(user_id, history_df, favorite_df)

        return jsonify({"status": "success", "recommendations": recommendations})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
