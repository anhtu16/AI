# ABSA Streamlit Dashboard – Nhóm 10

Hệ thống Aspect-based Sentiment Analysis phục vụ môn **Trí tuệ nhân tạo trong kinh doanh**. Ứng dụng chạy bằng Streamlit, tải hai mô hình đã fine-tune trên Google Colab (aspect multi-label + sentiment multi-class) và cung cấp giao diện hiện đại cho cả phân tích thủ công, phân tích file, dashboard và Action Center.

## 1. Yêu cầu môi trường

- Python 3.11 (đã kiểm thử với 3.11.2)
- Thư viện: `streamlit`, `transformers`, `torch`, `pandas`, `numpy`, `plotly`, `openpyxl` (đã liệt kê trong `absa_app/requirements.txt`)

## 2. Cài đặt và chạy

```bash
# 1. Tạo và kích hoạt virtualenv (nếu chưa)
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell dùng .\.venv\Scripts\activate

# 2. Cài thư viện
pip install -r absa_app/requirements.txt

# 3. Chạy Streamlit
cd absa_app
streamlit run app.py
```

Khi ứng dụng mở trên trình duyệt:

1. Vào tab **📁 Phân tích file** để upload CSV/XLS(X) (cột văn bản mặc định là `text` – có thể đổi).
2. Sau khi `Phân tích file`, chuyển sang **📊 Dashboard** để xem biểu đồ và **🎯 Action Center** để nhận gợi ý hành động.

## 3. Cấu trúc chính

```
absa_app/
├── app.py                 # Streamlit UI + logic
├── model_service.py       # Load model, inference, tổng hợp sentiment
├── models/
│   ├── aspect/            # model.safetensors, tokenizer.json, config chứa id2label
│   └── sentiment/         # model cảm xúc
├── requirements.txt
└── sample_reviews.csv     # Dataset mẫu demo
```

## 4. Tính năng nổi bật

- **🔍 Phân tích câu**: nhập một câu, chỉnh ngưỡng sigmoid, xem sentiment tổng thể và bảng aspect + sentiment tương ứng.
- **📁 Phân tích file**: upload CSV/Excel, chạy inference hàng loạt, tải kết quả CSV (bao gồm cột `aspects_detail` để phân tích sâu).
- **📊 Dashboard**: biểu đồ donut sentiment với gradient, biểu đồ tần suất aspect, line chart confidence theo review, stacked bar tỉ lệ sentiment theo aspect.
- **🎯 Action Center**: tổng hợp dữ liệu để đưa ra khuyến nghị thực tiễn:
  - Bảng khía cạnh cần ưu tiên xử lý (dựa trên tỉ lệ NEG và số lượng nhắc tới).
  - Bảng cơ hội nổi bật (aspect được khen nhiều).
  - Gợi ý hành động dạng bullet và trích ví dụ phản hồi tiêu biểu.
  - Nút tải báo cáo CSV phục vụ họp/triển khai.

## 5. Dataset mẫu

- `sample_reviews.csv`: 5 câu tiếng Việt dùng cho demo nhanh.
- Có thể tự tạo thêm file bằng cách giữ nguyên format `text` và upload trong tab phân tích file.

## 6. Ghi chú triển khai

- Để tránh lỗi cache, dùng `streamlit cache clear` mỗi khi thay đổi code hoặc mô hình.
- Nếu muốn đổi nhãn aspect, cập nhật `absa_app/models/aspect/config.json` (trường `id2label/label2id`) hoặc đặt `labels.json`.
- Mô hình sentiment đang nhận input theo định dạng `aspect: {ASPECT} text: {TEXT}` giống notebook gốc, nên inference khớp với kết quả Colab.

---

Made with ❤️ by Nhóm 10. Hände hoch AI!
