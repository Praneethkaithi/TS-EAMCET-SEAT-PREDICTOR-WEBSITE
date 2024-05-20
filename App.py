import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, render_template, send_file
from io import BytesIO
from fpdf import FPDF

app = Flask(__name__)

def load_and_process_data(file_path):
    # Load the data
    df = pd.read_excel(file_path)
    
    # Rename the column to match the expected column name
    if 'Branch_code' in df.columns:
        df.rename(columns={'Branch_code': 'Branch_\ncode'}, inplace=True)
    
    # Select relevant columns
    df = df[['Inst Code', 'Institution Name', 'Branch_\ncode', 'Rank']]
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    return df

def train_model(df):
    # Encode categorical features
    le_inst_code = LabelEncoder()
    le_inst_name = LabelEncoder()
    le_branch_code = LabelEncoder()
    
    df['Inst Code'] = le_inst_code.fit_transform(df['Inst Code'])
    df['Institution Name'] = le_inst_name.fit_transform(df['Institution Name'])
    df['Branch_\ncode'] = le_branch_code.fit_transform(df['Branch_\ncode'])
    
    # Prepare features and target
    X = df[['Rank']]
    y = df[['Inst Code', 'Institution Name', 'Branch_\ncode']]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model with adjusted parameters
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the encoders and model
    joblib.dump(le_inst_code, 'le_inst_code.pkl')
    joblib.dump(le_inst_name, 'le_inst_name.pkl')
    joblib.dump(le_branch_code, 'le_branch_code.pkl')
    joblib.dump(model, 'rank_predictor_model.pkl')
    
    return model, le_inst_code, le_inst_name, le_branch_code

def predict_institution_branch(rank, model, le_inst_code, le_inst_name, le_branch_code):
    # Make a prediction
    prediction = model.predict([[rank]])
    
    # Decode the prediction
    inst_code = le_inst_code.inverse_transform([prediction[0][0]])[0]
    inst_name = le_inst_name.inverse_transform([prediction[0][1]])[0]
    branch_code = le_branch_code.inverse_transform([prediction[0][2]])[0]
    
    return inst_code, inst_name, branch_code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    rank = int(request.form['rank'])

    # Load and process the data
    input_file = "C:/Users/admin/Desktop/input.xlsx"
    df = load_and_process_data(input_file)

    # Train the model
    model, le_inst_code, le_inst_name, le_branch_code = train_model(df)

    # Predictions
    inst_code, inst_name, branch_code = predict_institution_branch(rank, model, le_inst_code, le_inst_name, le_branch_code)

    # Generate PDF with predictions and return download link
    pdf_output = "F:/project/predictions.pdf"
    generate_pdf(rank, inst_code, inst_name, branch_code, pdf_output)

    return render_template('download.html')

def generate_pdf(rank, inst_code, inst_name, branch_code, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Prediction Results", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Rank: {rank}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Institution Code: {inst_code}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Institution Name: {inst_name}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Branch Code: {branch_code}", ln=True, align="L")
    pdf.output(output_path)

@app.route('/download')
def download():
    path = "F:/project/predictions.pdf"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
