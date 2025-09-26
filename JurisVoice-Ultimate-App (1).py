import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from collections import Counter
import re
import nltk
from textblob import TextBlob
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Page configuration
st.set_page_config(
    page_title="JurisVoice - AI Legal Feedback Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.4rem;
        margin-bottom: 3rem;
        font-weight: 500;
        animation: fadeInUp 1s ease-out 0.5s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Section Containers */
    .input-section {
        background: linear-gradient(135deg, #fef7cd 0%, #fbbf24 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(251, 191, 36, 0.3);
        border: 3px solid #f59e0b;
        position: relative;
        overflow: hidden;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #f59e0b, #d97706, #b45309);
    }
    
    .results-section {
        background: linear-gradient(135deg, #ecfdf5 0%, #a7f3d0 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.2);
        border: 3px solid #10b981;
        position: relative;
        overflow: hidden;
    }
    
    .results-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #10b981, #059669, #047857);
    }
    
    .qa-section {
        background: linear-gradient(135deg, #ede9fe 0%, #c4b5fd 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3);
        border: 3px solid #8b5cf6;
        position: relative;
        overflow: hidden;
    }
    
    .qa-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #8b5cf6, #7c3aed, #6d28d9);
    }
    
    .insights-section {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(236, 72, 153, 0.2);
        border: 3px solid #ec4899;
        position: relative;
        overflow: hidden;
    }
    
    .insights-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #ec4899, #db2777, #be185d);
    }
    
    .analytics-section {
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
        padding: 3rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(14, 165, 233, 0.2);
        border: 3px solid #0ea5e9;
        position: relative;
        overflow: hidden;
    }
    
    .analytics-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #0ea5e9, #0284c7, #0369a1);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.95);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8, #1e40af);
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.2);
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card p {
        margin: 10px 0;
        font-weight: 600;
        font-size: 1.1rem;
        color: #374151;
    }
    
    .metric-card small {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Insight Cards */
    .insight-card {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        font-size: 1.2rem;
        line-height: 1.8;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .insight-card:hover {
        transform: translateX(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .insight-card::before {
        content: '💡';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.5rem;
        opacity: 0.7;
    }
    
    /* Comment Input Boxes */
    .comment-box {
        background: rgba(255,255,255,0.95);
        border: 3px solid #e5e7eb;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .comment-box:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
        transform: scale(1.02);
    }
    
    /* Buttons */
    .primary-button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 1.5rem 4rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .primary-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: all 0.6s ease;
    }
    
    .primary-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.6);
    }
    
    .primary-button:hover::before {
        left: 100%;
    }
    
    .secondary-button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .secondary-button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.4);
    }
    
    /* Q&A Input */
    .qa-input {
        background: rgba(255,255,255,0.95);
        border: 3px solid #e5e7eb;
        border-radius: 15px;
        padding: 1.5rem;
        font-size: 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .qa-input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1);
        outline: none;
    }
    
    /* Progress Container */
    .progress-container {
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Animations */
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.8s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .slide-in-right {
        animation: slideInRight 0.8s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stDeployButton {display:none;}
    .stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'comments_list' not in st.session_state:
    st.session_state.comments_list = ['', '', '', '', '']
if 'num_comment_boxes' not in st.session_state:
    st.session_state.num_comment_boxes = 5
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

@st.cache_resource
def load_ai_models():
    """Load all AI models with enhanced caching"""
    try:
        with st.spinner("🤖 Loading Advanced AI Models..."):
            # Sentiment Analysis Model (RoBERTa)
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Text Summarization Model (BART)
            text_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=40,
                do_sample=False
            )
            
            # Question Answering Model (DistilBERT)
            qa_system = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            
        return sentiment_analyzer, text_summarizer, qa_system
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None, None

def preprocess_text(text):
    """Advanced text preprocessing"""
    if pd.isna(text) or not text or text.strip() == "":
        return ""
    
    text = str(text).strip()
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def analyze_sentiment_advanced(comments, sentiment_model):
    """Comprehensive sentiment analysis with multiple dimensions"""
    results = []
    
    for comment in comments:
        if not comment or pd.isna(comment) or comment.strip() == "":
            continue
            
        try:
            # Get sentiment scores from RoBERTa
            sentiment_scores = sentiment_model(comment[:512])
            
            # Process sentiment data
            if isinstance(sentiment_scores[0], list):
                sentiment_data = max(sentiment_scores[0], key=lambda x: x['score'])
            else:
                sentiment_data = sentiment_scores[0]
            
            # Map sentiment labels to standard format
            label_mapping = {
                'LABEL_0': 'negative', 'NEGATIVE': 'negative',
                'LABEL_1': 'neutral', 'NEUTRAL': 'neutral',
                'LABEL_2': 'positive', 'POSITIVE': 'positive'
            }
            
            sentiment = label_mapping.get(sentiment_data['label'], 'neutral')
            confidence = sentiment_data['score']
            
            # Enhanced analysis using TextBlob
            blob = TextBlob(comment)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Advanced emotion categorization
            if polarity > 0.3:
                emotion = 'very_supportive'
            elif polarity > 0.1:
                emotion = 'supportive'
            elif polarity > -0.1:
                emotion = 'neutral'
            elif polarity > -0.3:
                emotion = 'concerned'
            else:
                emotion = 'very_concerned'
            
            # Content complexity analysis
            word_count = len(comment.split())
            sentence_count = len([s for s in comment.split('.') if s.strip()])
            complexity = 'high' if word_count > 40 else 'medium' if word_count > 20 else 'low'
            
            # Key issue detection
            issues = []
            if any(word in comment.lower() for word in ['complex', 'difficult', 'hard', 'confusing']):
                issues.append('complexity')
            if any(word in comment.lower() for word in ['time', 'deadline', 'rush', 'quick']):
                issues.append('timeline')
            if any(word in comment.lower() for word in ['cost', 'expensive', 'burden', 'fee']):
                issues.append('cost')
            if any(word in comment.lower() for word in ['misuse', 'abuse', 'unfair']):
                issues.append('misuse_risk')
            
            results.append({
                'comment': comment,
                'sentiment': sentiment,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'emotion': emotion,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'complexity': complexity,
                'issues': issues
            })
            
        except Exception as e:
            st.warning(f"⚠️ Error analyzing comment: {str(e)[:100]}")
            continue
    
    return results

def generate_executive_summary(comments, summarizer):
    """Generate comprehensive AI summary using BART"""
    try:
        # Combine all comments
        combined_text = " ".join([str(comment) for comment in comments if comment])
        
        if len(combined_text) < 100:
            return "Insufficient content for comprehensive summary generation."
        
        # Process in chunks for better results
        chunk_size = 1000
        chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
        
        summaries = []
        for chunk in chunks[:3]:  # Process first 3 chunks
            if len(chunk.strip()) > 100:
                try:
                    summary = summarizer(chunk, max_length=130, min_length=40, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    continue
        
        final_summary = " ".join(summaries) if summaries else "The analysis reveals mixed feedback requiring balanced policy consideration."
        
        return final_summary
        
    except Exception as e:
        return f"Summary generation encountered an issue: {str(e)}"

def generate_word_cloud(comments):
    """Generate enhanced word cloud visualization"""
    try:
        # Combine all text
        text = " ".join([str(comment) for comment in comments if comment])
        
        if len(text) < 50:
            return None
        
        # Enhanced stopwords
        stop_words = set(stopwords.words('english'))
        stop_words.update([
            'law', 'draft', 'section', 'provision', 'will', 'would', 
            'could', 'should', 'one', 'also', 'said', 'say', 'get',
            'go', 'know', 'like', 'see', 'make', 'way', 'think'
        ])
        
        # Create word cloud with enhanced settings
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            stopwords=stop_words,
            max_words=120,
            colormap='plasma',
            relative_scaling=0.6,
            min_font_size=12,
            prefer_horizontal=0.7,
            collocations=False
        ).generate(text)
        
        return wordcloud
        
    except Exception as e:
        st.error(f"Word cloud generation error: {str(e)}")
        return None

def generate_people_insights(sentiment_data):
    """Generate advanced 'What People Want' statistical insights"""
    if not sentiment_data:
        return []
    
    df = pd.DataFrame(sentiment_data)
    total_comments = len(df)
    insights = []
    
    # Sentiment distribution insights
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    
    if sentiment_counts.get('negative', 0) > 30:
        insights.append(f"⚠️ {sentiment_counts['negative']:.0f}% express concerns about the proposed legislation")
    
    if sentiment_counts.get('positive', 0) > 25:
        insights.append(f"✅ {sentiment_counts['positive']:.0f}% show active support for the legal reforms")
    
    if sentiment_counts.get('neutral', 0) > 40:
        insights.append(f"⚖️ {sentiment_counts['neutral']:.0f}% maintain neutral stance, suggesting need for more information")
    
    # Emotion-based insights
    emotion_counts = df['emotion'].value_counts(normalize=True) * 100
    
    if emotion_counts.get('very_concerned', 0) > 15:
        insights.append(f"😰 {emotion_counts['very_concerned']:.0f}% express serious concerns requiring immediate attention")
    
    if emotion_counts.get('very_supportive', 0) > 20:
        insights.append(f"🎉 {emotion_counts['very_supportive']:.0f}% are very supportive and enthusiastic about changes")
    
    # Issue-specific insights
    all_issues = [issue for issues in df['issues'] for issue in issues]
    issue_counts = Counter(all_issues)
    
    if issue_counts.get('complexity') > 0:
        complexity_pct = (issue_counts['complexity'] / total_comments) * 100
        insights.append(f"🔍 {complexity_pct:.0f}% want simpler, more accessible language in legal documents")
    
    if issue_counts.get('timeline') > 0:
        timeline_pct = (issue_counts['timeline'] / total_comments) * 100
        insights.append(f"⏰ {timeline_pct:.0f}% highlight concerns about implementation timeline and deadlines")
    
    if issue_counts.get('cost') > 0:
        cost_pct = (issue_counts['cost'] / total_comments) * 100
        insights.append(f"💰 {cost_pct:.0f}% express concerns about financial impact and compliance costs")
    
    if issue_counts.get('misuse_risk') > 0:
        misuse_pct = (issue_counts['misuse_risk'] / total_comments) * 100
        insights.append(f"⚡ {misuse_pct:.0f}% highlight potential misuse risk and need for safeguards")
    
    # Content analysis insights
    high_complexity = df[df['complexity'] == 'high']
    if len(high_complexity) > total_comments * 0.3:
        insights.append(f"📝 {len(high_complexity)/total_comments*100:.0f}% provide detailed, comprehensive feedback indicating high engagement")
    
    # Confidence and subjectivity insights
    high_confidence = df[df['confidence'] > 0.8]
    if len(high_confidence) > total_comments * 0.6:
        insights.append(f"🎯 {len(high_confidence)/total_comments*100:.0f}% of AI predictions have high confidence, indicating clear sentiment patterns")
    
    highly_subjective = df[df['subjectivity'] > 0.7]
    if len(highly_subjective) > total_comments * 0.5:
        insights.append(f"💭 {len(highly_subjective)/total_comments*100:.0f}% express strong personal opinions rather than factual concerns")
    
    # Word count insights
    avg_words = df['word_count'].mean()
    if avg_words > 30:
        insights.append(f"📊 Average comment length is {avg_words:.0f} words, indicating thoughtful and detailed engagement")
    
    return insights[:10]  # Return top 10 insights

def interactive_qa_system(question, comments, qa_model):
    """Advanced Interactive Q&A system for instant answers"""
    try:
        # Combine all comments for context
        context = " ".join([str(comment) for comment in comments if comment])[:2500]
        
        if len(context) < 100:
            return "❌ Insufficient context available. Please provide more detailed comments for analysis."
        
        # Enhanced question processing
        question = question.strip()
        if not question.endswith('?'):
            question += "?"
        
        # Get answer from QA model
        result = qa_model(question=question, context=context)
        
        confidence = result['score']
        answer = result['answer']
        
        # Enhanced response formatting
        if confidence > 0.7:
            confidence_level = "Very High"
            emoji = "🎯"
        elif confidence > 0.5:
            confidence_level = "High"
            emoji = "✅"
        elif confidence > 0.3:
            confidence_level = "Moderate"
            emoji = "⚠️"
        else:
            confidence_level = "Low"
            emoji = "❓"
        
        formatted_answer = f"{emoji} **{answer}**\n\n*AI Confidence: {confidence_level} ({confidence:.1%})*"
        
        return formatted_answer
        
    except Exception as e:
        return f"❌ Q&A analysis error: {str(e)}"

def create_advanced_visualizations(sentiment_df):
    """Create comprehensive visualization suite"""
    
    # 1. Enhanced Sentiment Distribution Donut Chart
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    colors = ['#10b981', '#f59e0b', '#ef4444']
    
    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="📊 Overall Sentiment Distribution",
        color_discrete_sequence=colors,
        hole=0.5
    )
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14,
        pull=[0.1, 0.1, 0.1]
    )
    fig1.update_layout(
        title_font_size=20,
        title_x=0.5,
        font=dict(size=14),
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    # Add center text
    fig1.add_annotation(
        text=f"Total<br><b>{len(sentiment_df)}</b><br>Comments",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    # 2. Emotion Analysis Horizontal Bar Chart
    emotion_counts = sentiment_df['emotion'].value_counts()
    colors_emotion = px.colors.qualitative.Set3[:len(emotion_counts)]
    
    fig2 = px.bar(
        y=emotion_counts.index,
        x=emotion_counts.values,
        title="🎭 Detailed Emotional Response Analysis",
        color=emotion_counts.values,
        color_continuous_scale='Viridis',
        text=emotion_counts.values,
        orientation='h'
    )
    fig2.update_traces(texttemplate='%{text}', textposition='outside')
    fig2.update_layout(
        title_font_size=20,
        title_x=0.5,
        xaxis_title="Number of Comments",
        yaxis_title="Emotion Type",
        showlegend=False,
        height=450,
        font=dict(size=14)
    )
    
    # 3. Confidence vs Polarity Scatter Plot with Emotion
    fig3 = px.scatter(
        sentiment_df,
        x='confidence',
        y='polarity',
        color='sentiment',
        size='word_count',
        hover_data=['emotion', 'complexity'],
        title="🎯 AI Confidence vs Sentiment Polarity Analysis",
        color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'],
        size_max=20
    )
    fig3.update_layout(
        title_font_size=20,
        title_x=0.5,
        xaxis_title="AI Model Confidence Score",
        yaxis_title="Sentiment Polarity (-1 to +1)",
        height=450,
        font=dict(size=14)
    )
    
    # Add quadrant lines
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig3.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    
    # 4. Comment Complexity vs Engagement Analysis
    complexity_counts = sentiment_df['complexity'].value_counts()
    
    fig4 = px.sunburst(
        sentiment_df,
        path=['sentiment', 'complexity', 'emotion'],
        title="📝 Comment Complexity & Engagement Breakdown",
        color='word_count',
        color_continuous_scale='Blues'
    )
    fig4.update_layout(
        title_font_size=20,
        title_x=0.5,
        height=450,
        font=dict(size=14)
    )
    
    # 5. Issues Identification Chart
    all_issues = [issue for issues in sentiment_df['issues'] for issue in issues]
    if all_issues:
        issue_counts = Counter(all_issues)
        
        fig5 = px.bar(
            x=list(issue_counts.keys()),
            y=list(issue_counts.values()),
            title="⚡ Key Issues Identified in Feedback",
            color=list(issue_counts.values()),
            color_continuous_scale='Reds',
            text=list(issue_counts.values())
        )
        fig5.update_traces(texttemplate='%{text}', textposition='outside')
        fig5.update_layout(
            title_font_size=20,
            title_x=0.5,
            xaxis_title="Issue Type",
            yaxis_title="Frequency",
            showlegend=False,
            height=400,
            font=dict(size=14)
        )
    else:
        fig5 = None
    
    # 6. Word Count Distribution
    fig6 = px.histogram(
        sentiment_df,
        x='word_count',
        nbins=20,
        title="📊 Comment Length Distribution Analysis",
        color_discrete_sequence=['#3b82f6'],
        marginal="box"
    )
    fig6.update_layout(
        title_font_size=20,
        title_x=0.5,
        xaxis_title="Number of Words per Comment",
        yaxis_title="Frequency",
        height=400,
        font=dict(size=14)
    )
    
    return fig1, fig2, fig3, fig4, fig5, fig6

def main():
    # Enhanced Title and Subtitle
    st.markdown('<h1 class="main-title">⚖️ JurisVoice</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-Powered Legal Feedback Analyzer | Smart India Hackathon 2025</p>', unsafe_allow_html=True)
    
    # Load AI Models
    with st.spinner("🔄 Initializing Advanced AI Models..."):
        sentiment_model, summarizer, qa_model = load_ai_models()
    
    if not all([sentiment_model, summarizer, qa_model]):
        st.error("❌ Failed to load AI models. Please refresh the page and try again.")
        st.stop()
    
    st.success("✅ All AI models loaded successfully! Ready for analysis.")
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("# 🎛️ Control Panel")
        st.markdown("---")
        
        # Input method selection
        analysis_mode = st.radio(
            "📥 Choose Input Method:",
            ["📝 Manual Comment Entry", "📁 CSV File Upload", "🎯 Demo Dataset"],
            help="Select how you want to provide legal feedback for AI analysis"
        )
        
        st.markdown("---")
        
        # Feature highlights
        st.markdown("## 🚀 Core AI Features")
        
        feature_info = {
            "🤖 Interactive Q&A": "Ask natural questions about feedback and get instant AI-powered answers with confidence scores",
            "📊 People's Voice Insights": "Auto-generates statistics like '30% want simpler language' from feedback patterns",
            "📈 Advanced Analytics": "Multi-dimensional sentiment analysis with emotion detection and issue identification",
            "🎯 Smart Summarization": "AI-powered executive summaries using state-of-the-art BART models"
        }
        
        for feature, description in feature_info.items():
            with st.expander(feature):
                st.write(description)
        
        st.markdown("---")
        
        # Technical specifications
        st.markdown("## 🔧 AI Models")
        st.markdown("""
        - **Sentiment**: RoBERTa-base-sentiment  
        - **Summarization**: BART-large-CNN
        - **Q&A**: DistilBERT-SQUAD  
        - **NLP**: NLTK + TextBlob
        - **Visualization**: Plotly + Matplotlib
        """)
        
        st.markdown("---")
        st.markdown("### 🏆 SIH 2025")
        st.markdown("**Problem Statement:** 25035")
        st.markdown("**Team:** JurisVoice")
    
    # Input Section with Enhanced Interface
    st.markdown('<div class="input-section fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📝 Legal Comments Input Interface</div>', unsafe_allow_html=True)
    
    comments = []
    
    if analysis_mode == "📝 Manual Comment Entry":
        st.markdown("### 💬 Enter Legal Feedback Comments")
        st.markdown("*Provide detailed legal consultation feedback for comprehensive AI analysis*")
        
        # Comment management controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**Currently managing {st.session_state.num_comment_boxes} comment boxes**")
        
        with col2:
            if st.button("➕ Add Box", help="Add another comment input box"):
                st.session_state.num_comment_boxes += 1
                st.session_state.comments_list.append("")
                st.experimental_rerun()
        
        with col3:
            if st.session_state.num_comment_boxes > 1:
                if st.button("➖ Remove Box", help="Remove the last comment box"):
                    st.session_state.num_comment_boxes -= 1
                    if len(st.session_state.comments_list) > st.session_state.num_comment_boxes:
                        st.session_state.comments_list = st.session_state.comments_list[:st.session_state.num_comment_boxes]
                    st.experimental_rerun()
        
        with col4:
            if st.button("🗑️ Clear All", help="Clear all comment boxes"):
                st.session_state.comments_list = [""] * st.session_state.num_comment_boxes
                st.experimental_rerun()
        
        st.markdown("---")
        
        # Dynamic comment input boxes
        for i in range(st.session_state.num_comment_boxes):
            comment = st.text_area(
                f"💭 Legal Comment #{i+1}:",
                key=f"comment_input_{i}",
                height=140,
                placeholder=f"""Enter detailed legal feedback comment #{i+1} here...

Example: "The proposed section 12A needs clearer implementation guidelines for small businesses. The current language is too technical and may create compliance difficulties. I suggest adding more specific examples and extending the implementation timeline to 12 months."

Your detailed feedback helps improve policy making!""",
                help=f"Provide comprehensive legal feedback for comment {i+1}",
                value=st.session_state.comments_list[i] if i < len(st.session_state.comments_list) else ""
            )
            
            if comment.strip():
                # Ensure list is long enough
                while len(st.session_state.comments_list) <= i:
                    st.session_state.comments_list.append("")
                st.session_state.comments_list[i] = comment.strip()
        
        # Filter non-empty comments
        comments = [c for c in st.session_state.comments_list[:st.session_state.num_comment_boxes] if c.strip()]
        
        if comments:
            st.success(f"✅ {len(comments)} comments ready for AI analysis!")
            
            # Preview section
            with st.expander("👀 Preview Comments for Analysis"):
                for i, comment in enumerate(comments):
                    st.write(f"**Comment {i+1}:** {comment[:200]}{'...' if len(comment) > 200 else ''}")
        else:
            st.info("💡 Please enter at least one detailed comment to proceed with analysis.")
    
    elif analysis_mode == "📁 CSV File Upload":
        st.markdown("### 📂 Upload CSV File with Legal Comments")
        st.markdown("*Upload a CSV file where the first column contains legal feedback comments*")
        
        uploaded_file = st.file_uploader(
            "Choose your CSV file",
            type=['csv'],
            help="Upload a CSV file with legal comments in the first column. Supports any number of rows."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                comments = df.iloc[:, 0].dropna().astype(str).tolist()
                comments = [c.strip() for c in comments if c.strip()]
                
                st.success(f"✅ Successfully loaded **{len(comments)}** comments from uploaded file!")
                
                # File statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Valid Comments", len(comments))
                with col3:
                    avg_length = np.mean([len(c.split()) for c in comments])
                    st.metric("Avg Words/Comment", f"{avg_length:.1f}")
                
                # Preview uploaded comments
                with st.expander("📋 Preview Uploaded Comments"):
                    for i, comment in enumerate(comments[:10]):
                        st.write(f"**Row {i+1}:** {comment[:250]}{'...' if len(comment) > 250 else ''}")
                    if len(comments) > 10:
                        st.write(f"... and {len(comments) - 10} more comments")
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Please ensure your CSV file has comments in the first column.")
    
    else:  # Demo Dataset
        st.markdown("### 🎯 Comprehensive Legal Consultation Demo Dataset")
        st.markdown("*Using professionally crafted sample comments covering various legal feedback scenarios*")
        
        comments = [
            "This draft legislation is overly complex and needs significantly simpler language for common citizens to understand effectively. The technical jargon makes it inaccessible to most stakeholders who will be affected by these changes.",
            
            "I strongly support this amendment as it will dramatically improve transparency in corporate governance structures. This is exactly what the business community has been requesting for years and will boost investor confidence.",
            
            "The proposed section 12A contains ambiguous language that might be misused by regulatory authorities. Please reconsider this provision and add more specific safeguards to prevent potential abuse of power.",
            
            "Excellent initiative by MCA! This comprehensive reform will streamline business processes, reduce bureaucratic delays, and significantly improve India's ease of doing business ranking globally.",
            
            "I am deeply concerned about the excessive compliance burden this will place on small businesses. The costs and administrative requirements seem disproportionate. Please provide more exemptions for MSMEs.",
            
            "The implementation timeline of 6 months is unreasonably aggressive and unrealistic. Companies need at least 12-18 months to properly implement these changes, train staff, and update systems.",
            
            "This progressive legislation will substantially benefit the Indian economy, improve regulatory efficiency, and enhance our competitiveness in global markets. The reforms are long overdue and much needed.",
            
            "Several clauses in sections 15-18 are poorly drafted and ambiguous, which will inevitably lead to legal disputes and litigation. Please clarify the language and provide more detailed implementation guidelines.",
            
            "I genuinely appreciate the government's proactive effort to modernize these outdated regulatory frameworks. However, more extensive stakeholder consultation would have been beneficial before drafting.",
            
            "The financial penalties mentioned in section 22 seem disproportionately harsh for minor compliance violations. A graded penalty structure would be more fair and reasonable for businesses.",
            
            "More comprehensive public consultation is urgently needed before finalizing this important legislation. The current consultation period is insufficient for such complex regulatory changes.",
            
            "The highly technical language used throughout sections 5-8 is completely inaccessible to common citizens and small business owners who lack legal expertise. This defeats the purpose of public consultation.",
            
            "This amendment will effectively help reduce systemic corruption in administrative processes and improve governance transparency. The anti-corruption measures are particularly commendable.",
            
            "I strongly suggest adding more robust legal safeguards and oversight mechanisms to prevent potential misuse of the expanded regulatory powers granted to authorities under this legislation.",
            
            "The estimated implementation and compliance costs appear prohibitively expensive for small and medium enterprises. This could disproportionately impact smaller players and reduce market competition.",
            
            "Rural businesses and agricultural enterprises face unique operational challenges that this draft legislation doesn't adequately address. More sector-specific provisions are needed.",
            
            "Section 8 appears to directly contradict existing environmental protection laws and regulations. This inconsistency needs to be resolved before implementation to avoid legal conflicts.",
            
            "This appears to be a well-researched and thoughtfully crafted proposal that addresses many long-standing concerns raised by industry associations and legal experts over the years.",
            
            "The compliance requirements outlined in the annexure lack sufficient clarity and specificity, which will create confusion among businesses and potentially lead to unintentional violations.",
            
            "I recommend establishing dedicated government helpdesks and support centers to provide implementation guidance and assistance to businesses, especially smaller ones with limited resources.",
            
            "The draft fails to consider the unique challenges faced by startups and emerging businesses. Special provisions for new enterprises would encourage entrepreneurship and innovation.",
            
            "While the objectives are commendable, the execution seems rushed. A phased implementation approach would be more practical and allow for course corrections based on initial feedback."
        ]
        
        st.success(f"✅ Using comprehensive dataset with **{len(comments)}** professionally crafted legal comments")
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Comments", len(comments))
        with col2:
            avg_words = np.mean([len(c.split()) for c in comments])
            st.metric("📝 Avg Words", f"{avg_words:.0f}")
        with col3:
            total_words = sum([len(c.split()) for c in comments])
            st.metric("📚 Total Words", total_words)
        with col4:
            st.metric("🎯 Variety Score", "High")
        
        # Sample comments preview
        with st.expander("📋 Sample Comments Preview"):
            for i, comment in enumerate(comments[:5]):
                # Simple sentiment prediction for preview
                if any(word in comment.lower() for word in ['support', 'excellent', 'appreciate', 'commendable']):
                    preview_sentiment = "🟢 Positive"
                elif any(word in comment.lower() for word in ['concern', 'problem', 'harsh', 'difficult']):
                    preview_sentiment = "🔴 Negative"
                else:
                    preview_sentiment = "🟡 Neutral"
                
                st.write(f"**Sample {i+1}:** {comment}")
                st.caption(f"Preview Sentiment: {preview_sentiment} | Words: {len(comment.split())}")
                st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Execution Section
    if comments:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Enhanced analysis button
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            analyze_clicked = st.button(
                f"🚀 ANALYZE {len(comments)} COMMENTS WITH AI",
                key="main_analyze_button",
                help=f"Start comprehensive AI analysis of {len(comments)} legal comments using advanced models",
                type="primary"
            )
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem; color: #64748b; font-size: 1.1rem;">
                Ready to process <strong>{len(comments)} comments</strong> using<br>
                🤖 <strong>RoBERTa + BART + DistilBERT</strong> AI models
            </div>
            """, unsafe_allow_html=True)
        
        if analyze_clicked:
            # Enhanced progress tracking with detailed steps
            progress_container = st.container()
            
            with progress_container:
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                step_info = st.empty()
                
                try:
                    # Step 1: Preprocessing
                    status_text.markdown("### 🔄 Step 1/5: Advanced Text Preprocessing")
                    step_info.info("Cleaning and standardizing comment text for optimal AI analysis...")
                    progress_bar.progress(20)
                    
                    cleaned_comments = [preprocess_text(comment) for comment in comments]
                    cleaned_comments = [c for c in cleaned_comments if c.strip()]
                    
                    # Step 2: Sentiment Analysis
                    status_text.markdown("### 🧠 Step 2/5: Advanced Sentiment Analysis")
                    step_info.info("Running RoBERTa-based sentiment analysis with emotion detection...")
                    progress_bar.progress(40)
                    
                    sentiment_data = analyze_sentiment_advanced(cleaned_comments, sentiment_model)
                    
                    # Step 3: Text Summarization
                    status_text.markdown("### 📄 Step 3/5: Executive Summary Generation")
                    step_info.info("Generating comprehensive summary using BART neural network...")
                    progress_bar.progress(60)
                    
                    summary = generate_executive_summary(cleaned_comments, summarizer)
                    
                    # Step 4: Insights Generation
                    status_text.markdown("### 📊 Step 4/5: People's Voice Insights")
                    step_info.info("Generating 'What People Want' statistical insights...")
                    progress_bar.progress(80)
                    
                    insights = generate_people_insights(sentiment_data)
                    wordcloud = generate_word_cloud(cleaned_comments)
                    
                    # Step 5: Finalization
                    status_text.markdown("### ✨ Step 5/5: Finalizing Results")
                    step_info.info("Preparing interactive dashboard and visualizations...")
                    progress_bar.progress(100)
                    
                    # Store comprehensive results
                    st.session_state.analyzed_data = {
                        'sentiment_data': sentiment_data,
                        'summary': summary,
                        'insights': insights,
                        'wordcloud': wordcloud,
                        'comments': cleaned_comments,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'total_processed': len(sentiment_data)
                    }
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    step_info.empty()
                    
                    # Success celebration
                    st.balloons()
                    st.success("🎉 **Analysis Completed Successfully!** Scroll down to explore comprehensive results.")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    step_info.empty()
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Please try again or contact support if the issue persists.")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Results Display Section
    if st.session_state.analyzed_data:
        
        # Executive Summary Section
        st.markdown('<div class="results-section slide-in-left">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📄 AI-Generated Executive Summary</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-card" style="font-size: 1.3rem; line-height: 1.8;">
        <strong>🤖 Comprehensive AI Analysis Summary:</strong><br><br>
        {st.session_state.analyzed_data['summary']}
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis metadata
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.8); 
                    border-radius: 10px; font-size: 1rem; color: #64748b;">
        📊 Analysis completed on {st.session_state.analyzed_data['timestamp']} | 
        🎯 {st.session_state.analyzed_data['total_processed']} comments processed successfully
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Metrics Dashboard
        sentiment_df = pd.DataFrame(st.session_state.analyzed_data['sentiment_data'])
        
        st.markdown('<div class="results-section slide-in-right">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Key Performance Metrics Dashboard</div>', unsafe_allow_html=True)
        
        # Calculate comprehensive metrics
        total_comments = len(sentiment_df)
        positive_pct = (sentiment_df['sentiment'] == 'positive').mean() * 100
        negative_pct = (sentiment_df['sentiment'] == 'negative').mean() * 100
        neutral_pct = (sentiment_df['sentiment'] == 'neutral').mean() * 100
        avg_confidence = sentiment_df['confidence'].mean()
        avg_words = sentiment_df['word_count'].mean()
        
        # Enhanced metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>📈 {total_comments}</h2>
                <p>Total Comments</p>
                <small>Analyzed Successfully</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #10b981;">👍 {positive_pct:.1f}%</h2>
                <p>Positive Feedback</p>
                <small>Supportive Comments</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #ef4444;">👎 {negative_pct:.1f}%</h2>
                <p>Critical Feedback</p>
                <small>Concerns Raised</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #f59e0b;">⚖️ {neutral_pct:.1f}%</h2>
                <p>Neutral Stance</p>
                <small>Balanced Views</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #8b5cf6;">🎯 {avg_confidence:.2f}</h2>
                <p>AI Confidence</p>
                <small>Model Accuracy</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Automated "What People Want" Insights Section
        st.markdown('<div class="insights-section fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Automated "What People Want" Insights</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 5px solid #ec4899;">
        <strong>🎯 Innovation Feature:</strong> Our AI automatically analyzes feedback patterns and generates statistical insights 
        about what citizens and stakeholders actually want from this legislation. These insights help policymakers understand 
        public priorities and make data-driven decisions.
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.analyzed_data['insights']:
            for i, insight in enumerate(st.session_state.analyzed_data['insights']):
                st.markdown(f"""
                <div class="insight-card">
                    <strong>Key Insight #{i+1}:</strong><br>
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <strong>📊 Analysis Summary:</strong><br>
                The feedback demonstrates diverse stakeholder perspectives requiring balanced policy consideration. 
                More detailed comments would enable generation of specific statistical insights.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced Analytics Dashboard
        st.markdown('<div class="analytics-section fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Advanced Analytics Dashboard</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <strong>📈 Interactive Visualizations:</strong> Explore comprehensive charts and graphs that reveal patterns, 
        trends, and insights from the legal feedback analysis. Each visualization is powered by AI analysis results.
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Create comprehensive visualizations
            fig1, fig2, fig3, fig4, fig5, fig6 = create_advanced_visualizations(sentiment_df)
            
            # Display charts in organized layout
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Sentiment & Emotion", "🎯 Confidence & Analysis", "📝 Content & Issues"])
            
            with chart_tab1:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
                with chart_col2:
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            
            with chart_tab2:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
                with chart_col2:
                    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
            
            with chart_tab3:
                if fig5:
                    st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
                st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
        
        except Exception as e:
            st.error(f"❌ Chart generation error: {str(e)}")
        
        # Word Cloud Visualization
        st.markdown("### ☁️ Key Terms Word Cloud Analysis")
        
        if st.session_state.analyzed_data['wordcloud']:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <strong>🔍 Word Frequency Analysis:</strong> This word cloud highlights the most frequently mentioned terms 
            in the legal feedback, helping identify key themes and concerns.
            </div>
            """, unsafe_allow_html=True)
            
            fig_wc, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(st.session_state.analyzed_data['wordcloud'], interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Most Frequently Mentioned Terms in Legal Feedback', 
                        fontsize=18, fontweight='bold', pad=30)
            st.pyplot(fig_wc)
        else:
            st.warning("⚠️ Unable to generate word cloud - insufficient text variety in comments.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive Q&A Section
        st.markdown('<div class="qa-section slide-in-left">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🤖 Interactive Q&A Summarizer</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border: 2px solid #8b5cf6;">
        <strong>🎯 Innovation Feature - Interactive Q&A System:</strong><br><br>
        Ask natural language questions about the legal feedback and receive instant, contextual AI-generated answers. 
        This feature enables officials to quickly extract specific insights without manually reviewing all comments.
        Perfect for policy makers who need quick, data-driven answers to specific questions.
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced suggested questions
        st.markdown("### 💡 Quick Question Suggestions (Click to Ask)")
        
        suggestion_categories = {
            "General Analysis": [
                ("❓ What are the main complaints?", "What are the main complaints in the feedback?"),
                ("👍 What do people support most?", "What aspects do people support most in this legislation?"),
                ("⚠️ What are major concerns?", "What are the major concerns raised by stakeholders?"),
                ("📊 What changes do people want?", "What specific changes do people want to see?")
            ],
            "Specific Issues": [
                ("⏰ Timeline concerns?", "What concerns are raised about implementation timeline?"),
                ("💰 Cost-related issues?", "What cost-related concerns are mentioned in feedback?"),
                ("📝 Language complexity?", "Do people find the language too complex or difficult?"),
                ("🏢 Business impact?", "What business impacts are mentioned in the feedback?")
            ]
        }
        
        for category, suggestions in suggestion_categories.items():
            st.markdown(f"**{category}:**")
            cols = st.columns(len(suggestions))
            
            for col, (btn_text, question) in zip(cols, suggestions):
                with col:
                    if st.button(btn_text, key=f"suggest_{question[:20]}"):
                        st.session_state.current_question = question
        
        st.markdown("---")
        
        # Enhanced Q&A input interface
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "**🎙️ Ask Your Question About the Legal Feedback:**",
                value=st.session_state.current_question,
                placeholder="e.g., 'What percentage of people want simpler language?', 'What are the implementation concerns?', 'Which sections need revision?'",
                key="qa_question_input",
                help="Type any natural language question about the analyzed legal feedback"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            ask_clicked = st.button("🚀 Ask AI", type="primary", help="Get AI-powered answer with confidence scoring")
        
        if ask_clicked and question.strip():
            with st.spinner("🤖 AI is analyzing feedback and generating contextual answer..."):
                answer = interactive_qa_system(question, st.session_state.analyzed_data['comments'], qa_model)
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                # Clear current question
                st.session_state.current_question = ""
        
        # Enhanced Q&A History Display
        if st.session_state.qa_history:
            st.markdown("### 💬 Q&A Interaction History")
            st.markdown("*Recent questions and AI-generated answers with confidence scoring*")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history[-8:])):
                with st.expander(f"❓ **Q{len(st.session_state.qa_history)-i}:** {qa['question']} ⏰ *({qa['timestamp']})*", 
                               expanded=(i==0)):
                    st.markdown(qa['answer'])
                    
                    # Feedback buttons (UI only)
                    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 8])
                    with feedback_col1:
                        st.button("👍 Helpful", key=f"helpful_{i}_{qa['timestamp']}", 
                                help="Mark this answer as helpful")
                    with feedback_col2:
                        st.button("👎 Not Helpful", key=f"not_helpful_{i}_{qa['timestamp']}", 
                                help="Mark this answer as not helpful")
        else:
            st.info("💭 No questions asked yet. Try asking something specific about the legal feedback analysis!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Analysis Table
        st.markdown('<div class="results-section fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Detailed Comment-by-Comment Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
        <strong>📊 Comprehensive Analysis Table:</strong> Detailed breakdown of each comment with AI analysis scores, 
        sentiment classification, emotion detection, and complexity assessment. Use this data for in-depth review.
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced dataframe preparation
        display_df = sentiment_df[[
            'comment', 'sentiment', 'emotion', 'confidence', 
            'complexity', 'word_count', 'polarity'
        ]].copy()
        
        display_df['confidence'] = display_df['confidence'].round(3)
        display_df['polarity'] = display_df['polarity'].round(3)
        
        display_df.columns = [
            '💬 Comment Text', '😊 Sentiment', '🎭 Emotion', 
            '🎯 AI Confidence', '📊 Complexity', '📝 Word Count', '📈 Polarity Score'
        ]
        
        # Enhanced styling function
        def style_dataframe(df):
            def highlight_sentiment(val):
                if val == 'positive':
                    return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
                elif val == 'negative':
                    return 'background-color: #fee2e2; color: #991b1b; font-weight: bold;'
                else:
                    return 'background-color: #fef3c7; color: #92400e; font-weight: bold;'
            
            def highlight_confidence(val):
                if val > 0.8:
                    return 'background-color: #dbeafe; color: #1e40af; font-weight: bold;'
                elif val > 0.6:
                    return 'background-color: #f3f4f6; color: #374151;'
                else:
                    return 'background-color: #fef2f2; color: #991b1b;'
            
            return df.style.applymap(highlight_sentiment, subset=['😊 Sentiment']) \
                     .applymap(highlight_confidence, subset=['🎯 AI Confidence']) \
                     .format({'🎯 AI Confidence': '{:.3f}', '📈 Polarity Score': '{:.3f}'})
        
        # Display enhanced dataframe
        styled_df = style_dataframe(display_df)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=500,
            column_config={
                "💬 Comment Text": st.column_config.TextColumn(
                    "Comment",
                    help="Original legal feedback comment text",
                    max_chars=300
                ),
                "🎯 AI Confidence": st.column_config.ProgressColumn(
                    "AI Confidence",
                    help="Confidence level of AI prediction (0-1 scale)",
                    min_value=0,
                    max_value=1,
                ),
                "📝 Word Count": st.column_config.NumberColumn(
                    "Words",
                    help="Number of words in the comment",
                    format="%d"
                ),
                "📈 Polarity Score": st.column_config.NumberColumn(
                    "Polarity",
                    help="Sentiment polarity score (-1=Very Negative, +1=Very Positive)",
                    min_value=-1,
                    max_value=1,
                    format="%.3f"
                )
            }
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Export Section
        st.markdown('<div class="results-section slide-in-right">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💾 Export & Download Analysis Results</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <strong>📁 Multiple Export Formats:</strong> Download comprehensive analysis results in various formats 
        for further analysis, reporting, or integration with other systems. Choose the format that best suits your needs.
        </div>
        """, unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Enhanced CSV export
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Detailed CSV",
                data=csv_data,
                file_name=f"jurisvoice_detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Complete analysis data with all metrics and scores for each comment"
            )
        
        with export_col2:
            # Enhanced comprehensive report
            insights_text = '\n'.join([f'• {insight}' for insight in st.session_state.analyzed_data['insights']])
            qa_text = '\n'.join([f'Q: {qa["question"]}\nA: {qa["answer"]}\n' for qa in st.session_state.qa_history[-10:]])
            
            comprehensive_report = f"""
═══════════════════════════════════════════════════════════
              JurisVoice - AI Legal Feedback Analysis Report
═══════════════════════════════════════════════════════════

Generated: {st.session_state.analyzed_data['timestamp']}
Platform: Smart India Hackathon 2025 | Problem Statement 25035
Analysis Engine: Advanced AI (RoBERTa + BART + DistilBERT)

═══════════════════════════════════════════════════════════
                        EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════

{st.session_state.analyzed_data['summary']}

═══════════════════════════════════════════════════════════
                     KEY PERFORMANCE METRICS
═══════════════════════════════════════════════════════════

📊 Total Comments Analyzed: {total_comments}
👍 Positive Sentiment: {positive_pct:.1f}% ({int(positive_pct * total_comments / 100)} comments)
👎 Negative Sentiment: {negative_pct:.1f}% ({int(negative_pct * total_comments / 100)} comments)
⚖️ Neutral Sentiment: {neutral_pct:.1f}% ({int(neutral_pct * total_comments / 100)} comments)
🎯 Average AI Confidence: {avg_confidence:.3f} (Scale: 0-1)
📝 Average Words per Comment: {avg_words:.1f}

═══════════════════════════════════════════════════════════
                  AUTOMATED "WHAT PEOPLE WANT" INSIGHTS
═══════════════════════════════════════════════════════════

{insights_text if insights_text else "• Standard analysis completed with mixed sentiment distribution"}

═══════════════════════════════════════════════════════════
                    INTERACTIVE Q&A HISTORY
═══════════════════════════════════════════════════════════

{qa_text if qa_text else "No questions were asked during this analysis session."}

═══════════════════════════════════════════════════════════
                     TECHNICAL SPECIFICATIONS
═══════════════════════════════════════════════════════════

🤖 AI Models Used:
• Sentiment Analysis: RoBERTa-base (Twitter-trained for robust social sentiment)
• Text Summarization: BART-large-CNN (Facebook's state-of-the-art summarization)
• Question Answering: DistilBERT-SQUAD (Optimized for fast, accurate Q&A)
• Text Processing: NLTK + TextBlob (Advanced natural language processing)

📊 Analysis Features:
• Multi-dimensional sentiment analysis with confidence scoring
• Emotion detection and categorization
• Content complexity assessment
• Issue identification and categorization
• Interactive Q&A with contextual answers
• Automated statistical insight generation

═══════════════════════════════════════════════════════════
                          END OF REPORT
═══════════════════════════════════════════════════════════

Report generated by JurisVoice AI | Smart India Hackathon 2025
For technical support or questions, please contact the development team.
            """
            
            st.download_button(
                label="📋 Download Full Report",
                data=comprehensive_report,
                file_name=f"jurisvoice_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                help="Executive summary with insights, metrics, and Q&A history"
            )
        
        with export_col3:
            # Enhanced JSON export with comprehensive data
            json_export_data = {
                'metadata': {
                    'analysis_timestamp': st.session_state.analyzed_data['timestamp'],
                    'total_comments_processed': st.session_state.analyzed_data['total_processed'],
                    'analysis_engine': 'JurisVoice AI v1.0',
                    'models_used': {
                        'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                        'summarization': 'facebook/bart-large-cnn',
                        'question_answering': 'distilbert-base-cased-distilled-squad'
                    }
                },
                'summary': {
                    'executive_summary': st.session_state.analyzed_data['summary'],
                    'key_metrics': {
                        'total_comments': total_comments,
                        'positive_percentage': round(positive_pct, 2),
                        'negative_percentage': round(negative_pct, 2),
                        'neutral_percentage': round(neutral_pct, 2),
                        'average_confidence': round(avg_confidence, 3),
                        'average_words_per_comment': round(avg_words, 1)
                    }
                },
                'insights': {
                    'automated_people_insights': st.session_state.analyzed_data['insights'],
                    'total_insights_generated': len(st.session_state.analyzed_data['insights'])
                },
                'qa_interactions': {
                    'total_questions_asked': len(st.session_state.qa_history),
                    'recent_qa_history': st.session_state.qa_history[-15:],  # Last 15 Q&As
                },
                'analysis_details': {
                    'sentiment_distribution': {
                        'positive_count': int(positive_pct * total_comments / 100),
                        'negative_count': int(negative_pct * total_comments / 100),
                        'neutral_count': int(neutral_pct * total_comments / 100)
                    },
                    'emotion_analysis': sentiment_df['emotion'].value_counts().to_dict(),
                    'complexity_breakdown': sentiment_df['complexity'].value_counts().to_dict()
                }
            }
            
            st.download_button(
                label="🔗 Download JSON Data",
                data=json.dumps(json_export_data, indent=2, ensure_ascii=False),
                file_name=f"jurisvoice_complete_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                help="Structured data format for system integration and further analysis"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Completion Celebration
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0; padding: 3rem; 
                    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
                    border-radius: 20px; border: 3px solid #10b981; 
                    box-shadow: 0 20px 40px rgba(16, 185, 129, 0.2);">
            <h2 style="color: #047857; margin: 0; font-size: 2.5rem;">🎉 Analysis Successfully Completed!</h2>
            <p style="margin: 1.5rem 0; color: #065f46; font-size: 1.3rem; line-height: 1.6;">
                Your legal feedback has been comprehensively analyzed using state-of-the-art AI models.<br>
                <strong>Ready for policy decision-making and stakeholder communication.</strong>
            </p>
            <div style="margin: 1.5rem 0; color: #6b7280; font-size: 1rem;">
                <strong>🤖 Powered by JurisVoice AI Engine</strong><br>
                Smart India Hackathon 2025 | Problem Statement 25035<br>
                Advanced Legal Technology Solutions
            </div>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px;">
                <strong>📊 Analysis Summary:</strong> {total_comments} comments • {positive_pct:.1f}% positive • 
                {negative_pct:.1f}% negative • {len(st.session_state.analyzed_data['insights'])} insights generated
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Enhanced Welcome Section
        st.markdown('<div class="results-section fade-in">', unsafe_allow_html=True)
        st.markdown("""
        # 🚀 Welcome to JurisVoice - Advanced AI Legal Feedback Analyzer
        
        **Transform legal consultation feedback into actionable insights using cutting-edge artificial intelligence.**
        
        ---
        
        ## 🎯 Revolutionary Innovation Features
        
        ### 🤖 **Interactive Q&A Summarizer**
        - **Natural Language Queries**: Ask questions like *"What are the main complaints?"* or *"What percentage want simpler language?"*
        - **Instant AI Responses**: Get contextual, confidence-scored answers without manual review
        - **Multi-Query Support**: Ask follow-up questions for deeper insights
        - **Smart Context Understanding**: AI maintains conversation context for better answers
        
        ### 📊 **Automated "What People Want" Insights**
        - **Statistical Auto-Generation**: Creates insights like *"30% want simpler language"*, *"15% highlight misuse risk"*
        - **Pattern Recognition**: Identifies what citizens actually need from legislation
        - **Quantified Priorities**: Transforms qualitative feedback into quantitative insights
        - **Issue Categorization**: Automatically detects concerns about timeline, cost, complexity, etc.
        
        ### 🔍 **Multi-Dimensional AI Analysis**
        - **Advanced Sentiment**: Beyond positive/negative → Supportive, concerned, satisfied, very_concerned
        - **Emotion Detection**: Understand the emotional undertone of public response
        - **Confidence Scoring**: Know exactly how certain AI predictions are
        - **Content Analysis**: Assess comment complexity, engagement level, and word patterns
        
        ---
        
        ## 🛠️ **Technical Excellence**
        
        ### 🧠 **AI Models**
        - **RoBERTa-base**: State-of-the-art sentiment analysis with social media training
        - **BART-large-CNN**: Facebook's advanced neural summarization model
        - **DistilBERT-SQUAD**: Optimized question-answering for fast, accurate responses
        - **NLTK + TextBlob**: Advanced natural language processing pipeline
        
        ### 📈 **Visualization Suite**
        - **Interactive Charts**: Plotly-powered dynamic visualizations
        - **Word Cloud Analysis**: Visual representation of key themes
        - **Multi-Tab Dashboard**: Organized display of different analysis aspects
        - **Real-time Updates**: Charts update instantly based on analysis results
        
        ### 💾 **Export Capabilities**
        - **CSV Data**: Complete analysis with all metrics and scores
        - **Executive Reports**: Professional summaries for decision-makers
        - **JSON Format**: Structured data for system integration
        - **Multi-format Support**: Choose the best format for your workflow
        
        ---
        
        ## 📋 **How to Get Started**
        
        ### 1. **📥 Choose Input Method**
        Select from the sidebar:
        - **Manual Entry**: Type comments directly with dynamic input boxes
        - **File Upload**: Upload CSV files with bulk feedback
        - **Demo Dataset**: Use professional sample comments for testing
        
        ### 2. **📝 Provide Feedback**
        - Add detailed legal consultation comments
        - Use the dynamic interface to manage multiple inputs
        - Preview your comments before analysis
        
        ### 3. **🚀 Analyze with AI**
        - Click the analysis button to start processing
        - Watch real-time progress with detailed step information
        - AI models work together for comprehensive results
        
        ### 4. **📊 Explore Results**
        - View executive summary and key metrics
        - Discover automated "What People Want" insights
        - Interact with advanced charts and visualizations
        
        ### 5. **🤖 Ask Questions**
        - Use the Interactive Q&A feature for specific insights
        - Get instant answers with confidence scoring
        - Build a history of questions and responses
        
        ### 6. **💾 Export Reports**
        - Download comprehensive analysis in multiple formats
        - Share results with stakeholders and decision-makers
        - Integrate data with existing government systems
        
        ---
        
        ## 🏆 **Why Choose JurisVoice?**
        
        ✅ **Government-Ready**: Purpose-built for official legal consultation processes  
        ✅ **Scalable Solution**: Handle thousands of comments with consistent quality  
        ✅ **Transparent AI**: All decisions include confidence scores and explanations  
        ✅ **User-Friendly**: No technical expertise required for operation  
        ✅ **Comprehensive Analysis**: Far beyond basic sentiment analysis  
        ✅ **Real-time Processing**: Instant results for time-sensitive decisions  
        ✅ **Integration-Ready**: Export formats compatible with government systems  
        
        ---
        
        ### 🎬 **Ready to revolutionize legal consultation analysis?**
        
        **👈 Select an input method from the sidebar to begin your AI-powered analysis journey!**
        
        *Experience the future of legal feedback analysis with JurisVoice - where artificial intelligence meets legal expertise.*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature demonstration
        with st.expander("🎯 See Live Demo Preview"):
            demo_col1, demo_col2 = st.columns(2)
            
            with demo_col1:
                st.markdown("""
                ### 📊 **Sample Insights Generated:**
                - ⚠️ *"65% express concerns about implementation timeline"*
                - 🔍 *"23% want simpler, more accessible language"*
                - 💰 *"18% highlight cost burden on small businesses"*
                - ⚡ *"12% mention potential misuse risk concerns"*
                """)
            
            with demo_col2:
                st.markdown("""
                ### 🤖 **Sample Q&A Interactions:**
                
                **Q:** *"What are people most worried about?"*  
                **A:** *"Implementation timeline and compliance costs (Confidence: 89%)"*
                
                **Q:** *"Do people support the changes?"*  
                **A:** *"Mixed response - 35% supportive, 40% concerned (Confidence: 76%)"*
                """)
            
            st.image("https://via.placeholder.com/1200x400/667eea/FFFFFF?text=JurisVoice+Advanced+AI+Dashboard+Preview+%7C+Interactive+Analytics+%26+Insights", 
                    caption="Advanced Analytics Dashboard - Interactive Charts, AI Insights & Q&A Interface")

if __name__ == "__main__":
    main()