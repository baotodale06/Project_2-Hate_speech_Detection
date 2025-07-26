import streamlit as st
import time
import random
import pandas as pd

st.set_page_config(
    page_title="Phân loại chủ đề LGBT+",
    page_icon="🏳️‍🌈",
    layout="centered",
    initial_sidebar_state="auto",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.markdown("""
<style>
    /* Font chữ chung cho toàn bộ ứng dụng */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Tiêu đề chính */
    .st-emotion-cache-10trblm {
        color: #FF4B4B; /* Màu đỏ hồng cho tiêu đề */
        font-weight: 700;
    }

    /* Nút bấm */
    .stButton>button {
        border: 2px solid #FF4B4B;
        border-radius: 20px;
        color: #FF4B4B;
        padding: 10px 24px;
        background-color: transparent;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #FF4B4B;
        color: white;
        border-color: #FF4B4B;
    }
    .stButton>button:active {
        background-color: #E03A3A !important;
        color: white !important;
        border-color: #E03A3A !important;
    }

    /* Khung nhập văn bản */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
    }

    /* Khung chứa kết quả */
    .result-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }

    /* Tiêu đề kết quả */
    .result-header {
        font-size: 24px;
        font-weight: 600;
        color: #333;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Style cho từng nhãn kết quả */
    .label-box {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
        transition: all 0.2s ease-in-out;
    }
    .label-box.predicted {
        background-color: #28a745; /* Màu xanh lá cho nhãn được dự đoán (1) */
        color: white;
        font-weight: bold;
    }
    .label-title {
        font-size: 20px;
        font-weight: 700;
    }
    .label-description {
        font-size: 14px;
        color: #6c757d;
    }
    .label-box.predicted .label-description {
        color: #f0f0f0;
    }
    .label-confidence {
        font-size: 12px;
        font-style: italic;
        margin-top: 5px;
    }

</style>
""", unsafe_allow_html=True)


def model_predict(text: str):
    if not text.strip():
        # Trả về 0 nếu không có input
        return {label: 0.0 for label in ["L", "G", "B", "T", "O", "NR"]}

   #Predict ở đây nha

    return labels

# --- UI ---

st.title("🏳️‍🌈 Ứng dụng phân loại chủ đề LGBT+")
st.markdown("Nhập một đoạn văn bản để phân tích và xác định các chủ đề liên quan đến cộng đồng LGBT+.")

text_input = st.text_area(
    "Nhập văn bản của bạn vào đây...",
    height=200,
    placeholder="Ví dụ: 'Một câu chuyện cảm động về hành trình của người chuyển giới...'"
)

submitted = st.button("Phân tích văn bản")

if submitted:
    if text_input.strip():
        with st.spinner('Đang phân tích, vui lòng chờ...'):
            predictions = model_predict(text_input)

        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown('<p class="result-header">Kết quả phân tích</p>', unsafe_allow_html=True)

        label_descriptions = {
            "L": "Lesbian (Đồng tính nữ)",
            "G": "Gay (Đồng tính nam)",
            "B": "Bisexual (Song tính)",
            "T": "Transgender (Chuyển giới)",
            "O": "Other (Chủ đề khác liên quan)",
            "NR": "Not Related (Không liên quan)"
        }

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        sorted_labels = sorted(predictions.items(), key=lambda item: item[1], reverse=True)

        for i, (label, confidence) in enumerate(sorted_labels):
            is_predicted = 1 if confidence > 0.5 else 0
            
            with cols[i % 3]:
                predicted_class = "predicted" if is_predicted else ""
                st.markdown(f"""
                <div class="label-box {predicted_class}">
                    <div class="label-title">{label}</div>
                    <div class="label-description">{label_descriptions.get(label, "")}</div>
                    <div class="label-confidence">Độ tin cậy: {confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Vui lòng nhập một đoạn văn bản để phân tích.")

st.markdown("---")
st.markdown("Một sản phẩm được tạo với ❤️ bằng Streamlit.")
