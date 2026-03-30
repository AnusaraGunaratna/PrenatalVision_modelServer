# PRENATAL_VISION MODEL SERVER 

1. Install required Python packages: pip install -r requirements.txt
2. Setup Model Weights:
   - Download the model weights zip file.
   - Extract and copy all .pt files into the "weights" folder inside the project.
   - Ensure the following files are present: crl_hybrid.pt, crl_pvnet.pt, crl_ldb.pt, crl_yolo8.pt, crl_yolo11.pt, nt_hybrid.pt, nt_pvnet.pt, nt_ldb.pt, nt_yolo8.pt, nt_yolo11.pt.
3. Configure Environment: Create a .env file and set the SECRET_KEY and API_KEY.
4. Run the server: python -m app.main.py

