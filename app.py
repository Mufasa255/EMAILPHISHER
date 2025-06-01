import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Model configuration
MODEL_NAME = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load model and tokenizer once at startup"""
    global model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()  # Set to evaluation mode
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_phishing(text):
    """
    Predict if email/URL is phishing or legitimate
    """
    global model, tokenizer
    
    if not text.strip():
        return "Please enter some text to analyze", {}, ""
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities
        probs = predictions[0].tolist()
        
        # Label mapping
        labels = {
            "Legitimate Email": probs[0],
            "Phishing URL": probs[1], 
            "Legitimate URL": probs[2],
            "Phishing Email": probs[3] if len(probs) > 3 else 0
        }
        
        # Find highest probability
        max_label = max(labels.items(), key=lambda x: x[1])
        prediction = max_label[0]
        confidence = max_label[1]
        
        # Create confidence bar data
        confidence_data = {label: prob for label, prob in labels.items()}
  
        # Risk assessment
        if "Phishing" in prediction:
            risk_level = "🚨 HIGH RISK - Potential Phishing Detected"
            risk_color = "red"
        else:
            risk_level = "✅ LOW RISK - Appears Legitimate"
            risk_color = "green"
        
        # Format result
        result = f"""
### {risk_level}
**Primary Classification:** {prediction}  
**Confidence:** {confidence:.1%}
        """
        
        return result, confidence_data, risk_color
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", {}, "orange"

# Load model at startup
print("Loading model...")
model_loaded = load_model()
if not model_loaded:
    print("Failed to load model!")

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Phishing Email & URL Detective",
    css="""
    .risk-high { color: #dc2626 !important; font-weight: bold; }
    .risk-low { color: #16a34a !important; font-weight: bold; }
    .main-container { max-width: 800px; margin: 0 auto; }
    """
) as demo:
    
    gr.Markdown("""
    # 🛡️ Phishing Detection System
    **Instantly detect phishing emails and malicious URLs using AI**
    
    Powered by DistilBERT • 99.58% Accuracy • Real-time Analysis
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📧 Email Content or URL",
                placeholder="Paste suspicious email content or URL here...",
                lines=8,
                max_lines=15
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze for Phishing",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Analysis Result")
            
            confidence_output = gr.Label(
                label="Confidence Breakdown",
                num_top_classes=4
            )
    
    # Example inputs
    gr.Markdown("### 📋 Try These Examples:")
    
    examples = [
        ["Dear User, Your account will be suspended! Click here immediately: http://fake-bank-login.com/urgent"],
        ["Hi Mufasa, Thanks for your email. The quarterly report is attached. Best regards, Simba"],
        ["URGENT: Verify your PayPal account now or lose access: https://paypal-security-verify.suspicious.com"],
        ["Meeting reminder: Project sync at 3 PM in conference room B. See you there!"]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=[result_output, confidence_output]
    )
    
    # Event handlers
    analyze_btn.click(
        fn=predict_phishing,
        inputs=input_text,
        outputs=[result_output, confidence_output, gr.State()]
    )
    
    input_text.submit(
        fn=predict_phishing,
        inputs=input_text,
        outputs=[result_output, confidence_output, gr.State()]
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ About This Tool and the team.
    - **Model:** DistilBERT fine-tuned for phishing detection
    - **Accuracy:** 99.58% on test dataset
    - **Speed:** Real-time analysis
    - **Privacy:** All processing happens locally, no data stored
    
    **⚠️ Disclaimer:** This tool is for educational purposes (Assignemnt) only, we currently hold no rights and responsibility to this tool. So please Always verify suspicious content through official channels.
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False
    )
