import numpy as np
import joblib
import os
import fitz  # PyMuPDF

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

# 검증 함수
def validate_model(model_path, pdf_folder_path, scaler_path):
    # iForest 모델 및 스케일러 로드
    iforest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 폴더 내 모든 PDF 파일 경로를 리스트에 추가
    pdf_paths = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path) if file.endswith('.pdf')]

    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}")  # PDF 파일 경로 출력
        bytes_from_pdf = extract_bytes_from_pdf(pdf_path)
        
        # 데이터 정규화
        byte_data_normalized = scaler.transform(bytes_from_pdf)

        # 모델을 사용한 예측
        predictions = iforest.predict(byte_data_normalized)

        # 예측 결과 출력: -1은 이상치, 1은 정상
        for i, pred in enumerate(predictions):
            if pred == -1:
              break
              
        return pred

# 모델 및 스케일러 경로 설정
model_path = '/content/drive/MyDrive/iforest_byte_model2.pkl'
scaler_path = '/content/drive/MyDrive/scaler2.pkl'
pdf_folder_path = '/content/drive/MyDrive/문서경호/malicious_pdf'  # 테스트용 PDF 경로 설정

# 모델 검증
validate_model(model_path, pdf_folder_path, scaler_path)
