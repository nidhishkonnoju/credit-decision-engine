Intelli-Credit Project Standards (2026 Guidelines)
1. Project Architecture (Strict 9-Step Pipeline)

You must strictly follow the 9-Step Microservice Architecture:

    Step 1 - Document Parsing: OCR Extraction (PaddleOCR) -> Financial Entity Recognition -> Financial Metrics Extraction.

    Step 2 - GST Fraud Detection: GSTR 2A/2B vs 3B Reconciliation -> Transaction Graph Construction (NetworkX) -> Circular Trading Detection -> Fraud Risk Signals.

    Step 3 - Bank Statement: Balance & Cashflow Analysis -> Transaction Flow Analysis -> Debt Servicing Indicators.

    Step 4 - Officer Insight: Financial Sentiment Analysis (FinBERT) -> Risk Keyword Detection -> Operational Capacity Signals.

    Step 5 - External: Live Web Crawling (DuckDuckGo) -> News & Litigation Detection -> Sector Risk Signals.

    Step 6 - Feature: Unified Feature Matrix (Delta Lake).

    Step 7 - Explainability: Feature Contribution Analysis (SHAP TreeExplainer) -> Key Risk Drivers.

    Step 8 - Decision: Credit Risk Evaluation (LightGBM) -> Approve/Review/Reject, Risk-Based Pricing, Loan Limit Recommendation.

    Step 9 - CAM Generation: Llama-3 (Groq API) generating Executive Summary, Five Cs Credit Analysis, Financial Highlights, Risk and Fraud Analysis, AI Recommendation -> Credit Appraisal Memo PDF.