cat > README.md << 'EOF'
# FRA Analyzer (SIH25190)

AI-powered Frequency Response Analysis (FRA) tool for diagnosing power transformer faults.

## 🚀 Features
- Upload FRA data in CSV format
- AI model predicts transformer condition (Healthy / Faulty)
- Visualization of uploaded frequency response
- Streamlit-based web interface

## 🛠 Installation (Run Locally)
```bash
git clone https://github.com/mehersourav49-maker/fra-analyzer.git
cd fra-analyzer
pip install -r requirements.txt
streamlit run app.py
