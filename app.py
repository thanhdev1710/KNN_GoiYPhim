# Import thư viện
from flask import Flask, jsonify, request
from supabase import create_client
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Kết nối đến Supabase
SUPABASE_URL = "https://kburxjzaitqoesdzkkiq.supabase.co"  # Thay bằng URL dự án của bạn
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtidXJ4anphaXRxb2VzZHpra2lxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxOTc1ODExMCwiZXhwIjoyMDM1MzM0MTEwfQ.Ek0dBEgg9sJshaHUSWaW00_C_mPwVuwdWDTPtTg5qfM"  # Thay bằng API key của bạn
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Hàm lấy dữ liệu từ Supabase
def fetch_data():
    """
    Lấy dữ liệu từ bảng `watch_history` và `favorite_movies` trong Supabase.
    """
    # Lấy lịch sử xem phim
    watch_history = supabase.table("movieViewingHistory").select("userId, slug").execute()
    history_df = pd.DataFrame(watch_history.data)

    # Lấy danh sách phim yêu thích
    favorite_movies = supabase.table("listMovieFavorite").select("userId, slug").execute()
    favorite_df = pd.DataFrame(favorite_movies.data)

    return history_df, favorite_df

# Hàm gợi ý phim sử dụng thuật toán KNN
def recommend_movies(user_id, history_df, favorite_df, k=1):
    """
    Gợi ý phim cho người dùng dựa trên lịch sử xem phim và phim yêu thích.
    Args:
        user_id (int): ID của người dùng cần gợi ý.
        history_df (DataFrame): Dữ liệu lịch sử xem phim.
        favorite_df (DataFrame): Dữ liệu phim yêu thích.
        k (int): Số người dùng gần nhất để so sánh.
    Returns:
        list: Danh sách các slug của phim được gợi ý.
    """
    # Tạo ma trận người dùng-phim (sử dụng slug làm cột)
    user_movie_matrix = history_df.pivot_table(index="userId", columns="slug", aggfunc="size", fill_value=0)

    # Huấn luyện mô hình KNN để tìm người dùng tương tự
    model = NearestNeighbors()
    model.fit(user_movie_matrix)

    # Tìm k+1 người dùng gần nhất (bao gồm chính người dùng)
    distances, indices = model.kneighbors([user_movie_matrix.loc[user_id]], n_neighbors=k + 1)
    similar_users = user_movie_matrix.index[indices.flatten()].tolist()

    # Loại bỏ chính người dùng khỏi danh sách
    similar_users.remove(user_id)

    # Tổng hợp các phim yêu thích từ người dùng tương tự
    recommended_movies = set()
    for similar_user in similar_users:
        user_favorites = favorite_df[favorite_df["userId"] == similar_user]["slug"].tolist()
        recommended_movies.update(user_favorites)

    # Lấy danh sách phim mà người dùng đã xem
    watched_movies = history_df[history_df["userId"] == user_id]["slug"].tolist()

    # Lấy danh sách phim yêu thích của người dùng hiện tại
    favorite_movies = favorite_df[favorite_df["userId"] == user_id]["slug"].tolist()

    # Kết hợp cả hai danh sách để loại bỏ
    excluded_movies = set(watched_movies + favorite_movies)

    # Loại bỏ các phim mà người dùng đã xem hoặc yêu thích khỏi danh sách gợi ý
    final_recommendations = list(recommended_movies - excluded_movies)

    return final_recommendations

# Tạo ứng dụng Flask
app = Flask(__name__)

# API gợi ý phim
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API nhận `user_id` qua query string và trả về danh sách các slug của phim được gợi ý.
    """
    try:
        # Lấy `user_id` từ query string
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return jsonify({"status": "error", "message": "Missing user_id parameter"}), 400
        # Lấy dữ liệu từ Supabase
        history_df, favorite_df = fetch_data()
        print(favorite_df)

        # Gợi ý phim
        recommendations = recommend_movies(user_id, history_df, favorite_df)

        # Trả về kết quả
        return jsonify({"status": "success", "recommendations": recommendations})
    except Exception as e:
        # Trả về lỗi nếu có vấn đề
        return jsonify({"status": "error", "message": str(e)}), 500


# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)