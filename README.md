---
license: apache-2.0
title: EmailGuard
sdk: gradio
emoji: ⚡
colorFrom: yellow
colorTo: purple
short_description: The only secure and rational email phishing detector
---

# EmailGuard:  AI-Powered Phishing Detection System

The only secure and rational email phishing detector using advanced DistilBERT architecture for multilabel classification of emails and URLs.

## Model Architecture

**Base Model:** DistilBERT (Distilled Bidirectional Encoder Representations from Transformers)
- **Task Type:** Multilabel sequence classification
- **Framework:** Hugging Face Transformers
- **Fine-tuning:** 3 epochs using Trainer API
- **Input Length:** Maximum 512 tokens with truncation
- **Output Classes:** 4-class multilabel classification

## Performance Metrics

- **Accuracy:** 99.58%
- **F1-Score:** 99.579
- **Precision:** 99.583  
- **Recall:** 99.58%

## Dataset

Trained on custom dataset `cybersectony/PhishingEmailDetectionv2.0` containing labeled emails and URLs classified as legitimate or phishing attempts.

## Classification Categories

1. **Legitimate Email** - Normal email communications
2. **Phishing URL** - Malicious web links
3. **Legitimate URL** - Safe web links
4. **Phishing Email** - Fraudulent email attempts

## Technical Implementation

The model uses softmax activation for probability distribution across classes, with the highest probability determining the primary classification. Input preprocessing includes tokenization with padding and truncation to maintain consistent input dimensions.

## 🚀 Getting Started

### Option 1: Use Online (Recommended)
**Try EmailGuard instantly - no installation required!**
1. Visit our live demo on Hugging Face Spaces
2. Paste your email content or suspicious URL
3. Click "Analyze for Phishing" 
4. Get instant results with confidence scores

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://huggingface.co/spaces/[your-username]/EmailGuard
cd EmailGuard

# Install dependencies
pip install gradio==5.0.1 transformers torch

# Run locally
python app.py
```

## 💡 How to Use EmailGuard

1. **Input:** Paste suspicious email content, URLs, or text messages
2. **Analyze:** Click the analyze button or press Enter
3. **Review:** Check the risk assessment and confidence breakdown
4. **Verify:** Always cross-check results through official channels

### Example Inputs to Test:
- Suspicious payment verification emails
- Unknown links from social media
- Urgent account security messages
- Prize/lottery notification emails

## 📋 Suggestions & Best Practices

**✅ Good Use Cases:**
- Educational cybersecurity training
- Academic research projects
- Initial screening of suspicious content
- Learning about phishing patterns

**⚠️ Important Limitations:**
- This is a prototype for academic purposes
- Not intended for production security systems
- Always verify through official channels
- Combine with human judgment and expertise

## 🤝 Contact & Support

**Questions? Feedback? Collaboration?**

📧 **Email:** kelvinbyabato92@gmail.com

We welcome:
- Academic collaboration inquiries
- Technical feedback and suggestions
- Bug reports and improvement ideas
- Research partnership opportunities

## 🎯 Take Action Now!

**Ready to test EmailGuard?**
1. **[Try the Live Demo →]** Start analyzing suspicious emails instantly
2. **[Fork on GitHub →]** Contribute to the open-source project
3. **[Share with Friends →]** Help others stay safe from phishing

**Stay Safe Online!** 🛡️

---

### Academic Disclaimer

**Date:** May 30, 2025

This application is developed as an academic project by University of Dar es Salaam students: _**Byabato, Emmaculata, Regina, Sandy, Gladness, Alvin, Dorcas, and Albert**_.

**Important Notice:** This tool is intended solely for educational and research purposes. The developers hold no rights, benefits, or responsibilities regarding its use. Users are strongly advised to exercise caution and not rely on this system as a direct security solution. This is a prototype for academic evaluation and should not replace professional cybersecurity tools or expert judgment. Always verify suspicious content through official channels and established security protocols.

---

*Built with ❤️ by University of Dar es Salaam's Computer Science and Engineering (CSE) students*
