import numpy as np
import joblib
import fitz  # PyMuPDF
from sklearn.preprocessing import StandardScaler

# PDF 파일에서 바이트 데이터 추출 (200KB 한정)
def extract_bytes_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        byte_data = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").encode('utf-8')
            byte_vector = np.frombuffer(text, dtype=np.uint8)
            if len(byte_vector) > 200000:
                byte_vector = byte_vector[:200000]  # 200KB로 자르기
            byte_vector = np.pad(byte_vector, (0, max(0, 200000 - len(byte_vector))), 'constant')
            byte_data.append(byte_vector)
    byte_data = np.array(byte_data)
    return byte_data

# 모델 사용 함수
def use_model(model_path, scaler_path, pdf_path):
    iforest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    bytes_from_pdf = extract_bytes_from_pdf(pdf_path)

    byte_data_normalized = scaler.transform(bytes_from_pdf)

    predictions = iforest.predict(byte_data_normalized)

    # 예측 결과 출력: -1은 이상치, 1은 정상
    for i, pred in enumerate(predictions):
        status = "정상" if pred == 1 else "이상치"
        print(f"Page {i + 1} of {pdf_path} is classified as {status}")

# 모델 및 스케일러 경로 설정
model_path = 'iforest_byte_model.pkl'
scaler_path = 'scaler.pkl'
pdf_path = '/content/drive/MyDrive/문서경호/test_pdf/test_file.pdf'  # 테스트할 PDF 경로 설정

# 모델 사용
use_model(model_path, scaler_path, pdf_path)
