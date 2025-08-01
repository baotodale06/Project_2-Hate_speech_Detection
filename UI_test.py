import streamlit as st


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
            predictions = [1,1,0,0,0]

        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown('<p class="result-header">Kết quả phân tích</p>', unsafe_allow_html=True)

        label_w_pred = {label:predictions[id] for id, label in enumerate("LGBTO")}

        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        i=0
        for label, prediction in label_w_pred.items():
            with cols[i % 5]:
                answer = "Có" if prediction else "Không"
                st.markdown(f"""
                <div class="label-box {label}">
                    <div class="label-title">{label}</div>
                    <div class="label-confidence">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
                i+=1

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Vui lòng nhập một đoạn văn bản để phân tích.")
